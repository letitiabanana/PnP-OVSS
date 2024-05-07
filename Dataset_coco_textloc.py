import torch.utils.data as data
from PIL import Image
import os
import os.path
from PIL import ImageOps
# from lavis.models import load_model_and_preprocess
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import json
from skimage import transform as skimage_transform
from scipy.ndimage import filters
from pathlib import Path
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, args, annFile, getClassName, cats, vis_processors_clip=None, transform=None, target_transform=None, device="cpu"):

        self.root = root
        self.coco = COCO(annFile)
        if args.in_the_wild:
            self.ids = ['14','12']
        else:
            self.ids = list(self.coco.imgs.keys())

        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.getClassName = getClassName
        self.cats = cats
        self.args = args
        self.vis_processors_clip = vis_processors_clip

    def Attmap_resize(self, img_shape, attMap):
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap = attMap / attMap.max()
        attMap = skimage_transform.resize(attMap, (img_shape), order=3, mode="constant")
        return attMap

    def blurring(self, att_resize, img_shape):
        att_resize = filters.gaussian_filter(att_resize, 0.04*max(img_shape))
        att_resize = att_resize - att_resize.min()
        att_resize = att_resize / att_resize.max()
        return att_resize

    def Unsupervised_pn_mask_for_clipsim(self, args, img, img_id, cats, vis_processors_clip):
        complete_img = np.transpose(vis_processors_clip["eval"](img), (2, 0, 1))  # saved and checked, no prob

        # complete_img = np.expand_dims(complete_img, axis=0)
        # print("148 img.shape", complete_img.shape, complete_img.shape[1:]) #(1, 3, 336, 336) (3, 336, 336)

        with open(os.path.join(
                args.save_path + f"/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter0/max_att_class/",
                f"maxattclass_{img_id}_head{args.prune_att_head}.json"), 'r') as file:
            maxattclass_dict = json.load(file)
        text_for_clip = maxattclass_dict[
            f"{img_id}_maxattcllass_top0_layer{args.max_att_block_num}_head{args.prune_att_head}"]

        if args.drop_iter == 1:
            topk_prediction_mask_dict = np.load(os.path.join(
                args.save_path + f"/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter0/topk_img_att_forclasses/",
                f"img_{img_id}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy"),
                allow_pickle=True).item()
            prediction_mask = topk_prediction_mask_dict["att_map"][0]  # top0 map
        # else:
        #     prediction_mask_dict = combine_drop_iter_withclipsim(args, img_id, cats)
        #     prediction_mask = prediction_mask_dict["att_map"]

        try:
            if np.isnan(prediction_mask).any():
                print("1148 img:", img_id, "total pixel vs nan pixels", prediction_mask.shape[0] * 48 * 48, "vs",
                      np.isnan(prediction_mask).sum())
                prediction_mask[np.isnan(prediction_mask)] = 0
                print("att map contains nan", img_id)
            elif prediction_mask==None:
                print("data text loc img:", img_id, "None type for prediction mask, image has no gt classes")
        except:
            print('np.isnan not working')
            print(type(prediction_mask), prediction_mask, img_id)
            return np.zeros((1,3,224,224)),["No gt class for the image"]

        # text_for_clip = gt_class_name  # [gt_class_name for _ in range(len(gt_class_name)*2)]

        prediction_mask_pos = prediction_mask.copy()  # predmask_beforetrans checked, np

        prediction_mask_pos = (prediction_mask_pos > args.final_att_threshold).astype(int)

        prediction_mask_pos = self.Attmap_resize(complete_img.shape[-2:], prediction_mask_pos * 255)

        positive_mask = np.multiply(prediction_mask_pos, complete_img)  # check dim

        if img_id in [86483, 312213, 445248, 205105, 266981, 268831, 271471, 263796, 481480,
                      153343, 92091, 483050, 509258, 437351, 312278, 267537, 205282, 443303,
                      438017, 455157, 540414, 519764, 15278, 106563, 314294]:
            save_edge_map = positive_mask
            save_edge_map[save_edge_map > 255] = 255
            # print("1280",save_edge_map.max(), save_edge_map.min(), save_edge_map.shape)
            im = Image.fromarray(np.transpose(save_edge_map, (1, 2, 0)).astype(np.uint8))
            im.save(
                f"./Pos_mask/pos_img{img_id}_top0class{text_for_clip}_{args.max_att_block_num}_{args.prune_att_head}_attthresh{args.final_att_threshold}.jpeg")


        final_input_clip = torch.from_numpy(positive_mask).type('torch.FloatTensor')


        return final_input_clip, text_for_clip

    def token_cos_sim_map(self, img_ids, inputs, layer, head,
                          with_att_threshold=None):  # 算的是本身的token contrast，暂时与sort_threshold无关
        Path(
            f"Token_Contrast/max_att_block_num{layer}_atthead{head}_withatt{with_att_threshold}/Cos_sim_map/").mkdir(
            parents=True, exist_ok=True)
        Path(
            f"Token_Contrast/max_att_block_num{layer}_atthead{head}_withatt{with_att_threshold}/Token_contrast_sum/").mkdir(
            parents=True, exist_ok=True)
        import torch.nn.functional as F
        # print("322 ", inputs.shape)
        b, token_len, h, w = inputs.shape

        inputs = inputs.reshape(b, token_len, h * w)

        def cos_sim(x):
            x = F.normalize(x, p=2, dim=1, eps=1e-8)
            sim = torch.matmul(x.transpose(1, 2), x)
            # return torch.abs(cos_sim) # why abs???
            return sim

        inputs_cossim = cos_sim(inputs)
        # print("331 Blip image text matching inputs_cos", inputs_cos.shape, inputs_cos.max(), inputs_cos.min(), inputs_cos[0,].detach().cpu().numpy().shape)
        sum_std_dict = {}
        for i in range(inputs_cossim.shape[0]):
            by_img_att = inputs_cossim[i,].detach().cpu().numpy()
            if img_ids[i] in [397133, 86483, 312213, 445248, 205105, 266981, 268831, 271471, 263796, 481480,
                              153343, 92091, 483050, 509258, 437351, 312278, 267537, 205282, 443303,
                              438017, 455157, 540414, 519764, 15278, 106563, 314294]:
                cos_sim_map = Image.fromarray((by_img_att * 255).astype(np.uint8), 'L')
                cos_sim_map.save(
                    f"Token_Contrast/max_att_block_num{layer}_atthead{head}_withatt{with_att_threshold}/Cos_sim_map/img_{img_ids[i]}.jpeg")
            std = np.std(by_img_att)
            sum_token_sim = np.sum(by_img_att) / (by_img_att.shape[0] * by_img_att.shape[1])
            sum_std_dict[f"TCSum_layer{layer}_head{head}_img_{img_ids[i]}"] = sum_token_sim
            sum_std_dict[f"TCStd_layer{layer}_head{head}_img_{img_ids[i]}"] = std
            sum_std_dict_path = f"Token_Contrast/max_att_block_num{layer}_atthead{head}_withatt{with_att_threshold}/Token_contrast_sum/img_{img_ids[i]}.json"

            with open(sum_std_dict_path, "w") as outfile:
                json.dump(sum_std_dict, outfile, cls=NumpyArrayEncoder)

    def drop_image_patch_with_highest_att_eval(self, args, drop_iter, att, img_id, del_patch_num):

        # For now, drop simultaneously the highest att patch for all the classes of an image
        for drop in range(drop_iter):  # range start from 0 already
            # print("255 drop iter", drop_iter)
            att_loss_path = f"{args.save_path}/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop}/highest_att_save/highest_att_{img_id}_{args.prune_att_head}.json"
            with open(att_loss_path, 'r') as file:
                att_loss_dict = json.load(file)

            # for class_idx, class_name in enumerate(gt_class_name):
            try:
                # highest_att_idx = att_loss_dict[f"{args.max_att_block_num}_{img_id}_{class_name}"]
                highest_att_idx = att_loss_dict[f"halving_{img_id}"]
                if isinstance(args.del_patch_num, int):
                    max_patchs = highest_att_idx[-del_patch_num:]
                elif "sort_thresh" in args.del_patch_num:
                    max_patchs = highest_att_idx
            except:
                print(f"{args.max_att_block_num}_{img_id}_{class_name}", att_loss_path)
                print("217 highest att sequence of this image do not exist")
                breakpoint()

            # print("220 max_patch", drop_iter, img_id, class_name, max_patchs)
            patch_num = int(768 / 16)
            for max_patch in max_patchs:
                # print("44 img[max_x:max_x+16, max_y:max_y+16] before", att.sum())
                # max_x = (max_patch // 24) * 16
                # max_y = (max_patch % 24) * 16
                # print("img[max_x:max_x+16, max_y:max_y+16] before", max_x, max_x + 16, max_y, max_y + 16, img0[:, max_x:max_x + 16, max_y:max_y + 16].sum(), norm_img0[max_x:max_x + 16, max_y:max_y + 16, :].sum(), img0.sum())
                # att[:, max_x:max_x + 16, max_y:max_y + 16] = 0
                # print("img[max_x:max_x+16, max_y:max_y+16]", max_x, max_x + 16, max_y, max_y + 16, img0[:, max_x:max_x + 16, max_y:max_y + 16].sum(), norm_img0[max_x:max_x + 16, max_y:max_y + 16, :].sum(), img0.sum())
                max_x = max_patch // patch_num
                max_y = max_patch % patch_num

                att[:, max_x, max_y] = 0
                # print("55 img[max_x:max_x+16, max_y:max_y+16] after", max_x, max_y, att.sum())

        return att

    def densecrf(self, image, mask):
        MAX_ITER = 20
        POS_W = 7
        POS_XY_STD = 3
        Bi_W = 10
        Bi_XY_STD = 50
        Bi_RGB_STD = 5

        # h, w = mask.shape
        # mask = mask.reshape(1, h, w)
        # fg = mask.astype(float)
        # bg = 1 - fg
        # output_logits = torch.from_numpy(np.concatenate((bg, fg), axis=0))
        output_logits = mask

        H, W = image.shape[:2]
        # image = skimage_transform.resize(image, (image.shape[0] * 2, image.shape[1] * 2, image.shape[2]), order=3,
        #                                  mode="constant")
        image = np.ascontiguousarray(image.copy())
        # for i in range(image.shape[2])

        output_probs = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear").squeeze().cpu().numpy()
        # output_probs = F.softmax(output_logits, dim=0).cpu().numpy()
        if len(output_probs.shape) < 3:
            output_probs = np.expand_dims(output_probs, axis=0)

        try:
            c = output_probs.shape[0]
            h = output_probs.shape[1]
            w = output_probs.shape[2]
        except:
            breakpoint()

        U = utils.unary_from_softmax(output_probs)
        U = np.ascontiguousarray(U)

        d = dcrf.DenseCRF2D(w, h, c)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
        d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

        Q = d.inference(MAX_ITER)
        Q = np.array(Q).reshape((c, h, w))
        MAP = np.argmax(Q, axis=0).reshape((h, w)).astype(np.float32)
        print(MAP.shape)
        return MAP

    def combine_drop_iter_withclipsim(self, args, img_id, gt_shape):
        att_by_dropiter_list = []
        cam_dict_drop = {}
        for drop_iter in range(args.drop_iter):
            cam_dict_by_dropiter = np.load(os.path.join(
                args.save_path + f"/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter0/img_att_forclasses/",
                f"img_{img_id}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy"),
                allow_pickle=True).item()
            if cam_dict_by_dropiter["att_map"] is not None:
                if np.isnan(cam_dict_by_dropiter["att_map"]).any():
                    cam_dict_by_dropiter["att_map"][
                        np.isnan(cam_dict_by_dropiter["att_map"])] = 0
                    print("att map contains nan", id, drop_iter)

            if drop_iter > 0 and att_by_dropiter_list[0] is not None:
                cam_dict_by_dropiter["att_map"] = self.drop_image_patch_with_highest_att_eval(args, drop_iter,
                                                                                             cam_dict_by_dropiter[
                                                                                                 "att_map"], img_id,
                                                                                             args.del_patch_num)
            att_by_dropiter_list.append(cam_dict_by_dropiter["att_map"])

        if att_by_dropiter_list[0] is None:
            # print("524 len(att_by_dropiter_list)", id, len(att_by_dropiter_list))
            print("img has no cam", img_id)
            cam_dict_drop = {"att_map": None, "keys": None}
        else:
            weighted_droppatchatt = att_by_dropiter_list[0]
            for iter in range(args.drop_iter):  # for each drop iter
                weighted_droppatchatt += att_by_dropiter_list[iter]

            '''Thresholding the attention map'''
            for cls_idx in range(weighted_droppatchatt.shape[0]):
                weighted_droppatchatt[cls_idx][weighted_droppatchatt[cls_idx] < 0.15] = 0  #
                weighted_droppatchatt[cls_idx][weighted_droppatchatt[cls_idx] >= 0.15] = 1  #

            '''Apply CRF for mask refinement before threshold'''

            cams_resize = []
            for i in range(weighted_droppatchatt.shape[0]):
                att_resize = self.Attmap_resize(gt_shape, weighted_droppatchatt[i])

                att_resize = filters.gaussian_filter(att_resize, self.args.blur_std * max(gt_shape))
                att_resize -= att_resize.min()
                att_resize /= att_resize.max()
                cams_resize.append(att_resize)
            weighted_droppatchatt = np.stack(cams_resize, axis=0)

            if args.data_type == "cocoall":
                # cocoall already include annotation for almost everything in the picture, background are actully foreground under this setting
                pass
            else:
                background = 1 - weighted_droppatchatt.sum(axis=0)
                background[background < 0] = 0
                weighted_droppatchatt = np.concatenate(
                    (np.expand_dims(background, axis=0), weighted_droppatchatt), axis=0)

            path = self.coco.loadImgs(img_id)[0]['file_name']

            org_image = Image.open(os.path.join(self.root, path)).convert('RGB')

            org_image = np.asarray(org_image)
            weighted_droppatchatt_T = torch.from_numpy(weighted_droppatchatt)
            if len(weighted_droppatchatt_T.shape) < 3:
                weighted_droppatchatt_T = weighted_droppatchatt_T.unsqueeze(0)
            map_b = self.densecrf(org_image, weighted_droppatchatt_T)
            final_weighted_droppatchatt_list = []
            for i in range(weighted_droppatchatt.shape[0]):
                if args.data_type == "cocoall":
                    a = map_b == i
                    # a = skimage_transform.resize(a, (gt.shape[0], gt.shape[1]))
                    final_weighted_droppatchatt_list.append(a.astype(int))
                else:
                    if i > 0:
                        a = map_b == i
                        # a = skimage_transform.resize(a, (gt.shape[0], gt.shape[1]))
                        final_weighted_droppatchatt_list.append(a.astype(int))
            weighted_droppatchatt = np.stack(final_weighted_droppatchatt_list)
            cam_dict_drop["att_map"] = weighted_droppatchatt
            cam_dict_drop["keys"] = cam_dict_by_dropiter["keys"]
        return cam_dict_drop
    def Wsupervised_pn_mask_for_allimgs(self, args, img, img_id, vis_processors_clip, cats):

        complete_img = np.transpose(vis_processors_clip["eval"](img), (2, 0, 1))  # saved and checked, no prob

        complete_img = np.expand_dims(complete_img, axis=0)
        # print("148 img.shape", complete_img.shape, complete_img.shape[1:]) #(1, 3, 336, 336) (3, 336, 336)

        if args.drop_iter == 1:
            prediction_mask_dict = np.load(os.path.join(
                args.save_path + f"/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter0/img_att_forclasses/",
                f"img_{img_id}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy"),
                allow_pickle=True).item()
            prediction_mask = prediction_mask_dict["att_map"]

        else:
            prediction_mask_dict = self.combine_drop_iter_withclipsim(args, img_id, complete_img.shape[-2:])
            prediction_mask = prediction_mask_dict["att_map"]

        # print("1212", prediction_mask.max(),prediction_mask.min())
        gt_class_name = []
        for key in prediction_mask_dict["keys"]:
            gt_class_name.append(self.getClassName(key, cats))

        try:
            if np.isnan(prediction_mask).any():
                print("1148 img:", img_id, "total pixel vs nan pixels", prediction_mask.shape[0] * 48 * 48, "vs",
                      np.isnan(prediction_mask).sum())
                prediction_mask[np.isnan(prediction_mask)] = 0
                print("att map contains nan", img_id)
            elif prediction_mask is None:
                print("data text loc img:", img_id, "None type for prediction mask, image has no gt classes")
        except:
            print('np.isnan not working')
            print(type(prediction_mask), prediction_mask, img_id, gt_class_name)
            return np.zeros((1,3,224,224)),["No gt class for the image"]

        # predmask_beforetrans checked, np

        text_for_clip = gt_class_name  # [gt_class_name for _ in range(len(gt_class_name)*2)]

        prediction_mask_pos = prediction_mask.copy()  # predmask_beforetrans checked, np

        prediction_mask_pos = (prediction_mask_pos > args.final_att_threshold).astype(int)
        list_of_pms = []
        # list_of_nms = []
        list_of_token_pms = []
        for i in range(prediction_mask_pos.shape[0]):
            pms = self.Attmap_resize(complete_img.shape[-2:], prediction_mask_pos[i])
            pms = self.blurring(pms, complete_img.shape[-2:])
            # nms = self.Attmap_resize(complete_img.shape[-2:], 1-prediction_mask_pos[i])
            # nms = self.blurring(nms, complete_img.shape[-2:])
            token_pms = self.Attmap_resize((24,24 ), pms)
            list_of_pms.append(pms)
            # list_of_nms.append(nms)
            list_of_token_pms.append(token_pms)
        prediction_mask_pos = np.stack(list_of_pms, axis=0)
        # prediction_mask_neg = np.stack(list_of_nms, axis=0)
        prediction_mask_pos_token = np.stack(list_of_token_pms, axis=0)
        '''calculate token contrast'''
        self.token_cos_sim_map([img_id], torch.from_numpy(np.expand_dims(prediction_mask_pos_token, axis=0)), args.max_att_block_num,
                          args.prune_att_head, with_att_threshold=int(args.final_att_threshold * 100))

        # prediction_mask_neg = 1 - prediction_mask_pos
        prediction_mask_neg = np.zeros(prediction_mask_pos.shape)
        assert prediction_mask_pos.shape == prediction_mask_neg.shape
        # print("241 np.sum(negative_mask)", np.sum(prediction_mask_neg))
        prediction_mask_list_pos = []
        prediction_mask_list_neg = []
        # for i in range(prediction_mask_pos.shape[0]):
        #     # print("1249", complete_img.shape[2:], prediction_mask_pos[i, :].shape, skimage_transform.resize(prediction_mask_pos[i, :], complete_img.shape[2:], order=0, mode="constant").shape)
        #     prediction_mask_list_pos.append(
        #         np.transpose(np.repeat(
        #             np.expand_dims(self.Attmap_resize(complete_img.shape[2:], prediction_mask_pos[i, :]), axis=0), 3,
        #             0),
        #             (0, 1, 2)))
        #     prediction_mask_list_neg.append(
        #         np.transpose(np.repeat(
        #             np.expand_dims(self.Attmap_resize(complete_img.shape[2:], prediction_mask_neg[i, :] ), axis=0), 3,
        #             0),
        #             (0, 1, 2)))

        for i in range(prediction_mask_pos.shape[0]):
            # print("1249", complete_img.shape[2:], prediction_mask_pos[i, :].shape, skimage_transform.resize(prediction_mask_pos[i, :], complete_img.shape[2:], order=0, mode="constant").shape)
            prediction_mask_list_pos.append(
                np.transpose(np.repeat(np.expand_dims(prediction_mask_pos[i, :], axis=0), 3,0),
                    (0, 1, 2)))
            prediction_mask_list_neg.append(
                np.transpose(np.repeat(
                    np.expand_dims(prediction_mask_neg[i, :], axis=0), 3,
                    0),
                    (0, 1, 2)))


        prediction_mask_pos = np.stack(prediction_mask_list_pos)
        prediction_mask_neg = np.stack(prediction_mask_list_neg)
        # print("260 np.sum(negative_mask)", np.sum(prediction_mask_neg))
        prediction_mask_pos = (prediction_mask_pos > args.final_att_threshold).astype(int) # added oct 10th

        positive_mask = np.multiply(prediction_mask_pos,
                                    np.repeat(complete_img, len(gt_class_name), axis=0))  # check dim
        negative_mask = np.multiply(prediction_mask_neg,
                                    np.repeat(complete_img, len(gt_class_name), axis=0))  # check dim
        # print("274 np.sum(negative_mask)", np.sum(negative_mask))
        # print("1278 positive_mask.shape", positive_mask.shape, np.repeat(edge_map,3,0).shape, (positive_mask[0,:] + np.repeat(edge_map,3,0)).shape)
        assert len(positive_mask.shape) > 3
        if img_id in [397133, 86483, 312213, 445248, 205105, 266981, 268831, 271471, 263796, 481480,
                      153343, 92091, 483050, 509258, 437351, 312278, 267537, 205282, 443303,
                      438017, 455157, 540414, 519764, 15278, 106563, 314294]:
            for i in range(len(gt_class_name)):
                save_edge_map = positive_mask[i, :]
                save_edge_map[save_edge_map > 255] = 255
                # print("1280",save_edge_map.max(), save_edge_map.min(), save_edge_map.shape)
                im = Image.fromarray(np.transpose(save_edge_map, (1, 2, 0)).astype(np.uint8))
                # print("data text loc 207", f"./Edge_plus_pos/pos_img{img_id}_class{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}_att{args.final_att_threshold}.jpeg")
                im.save(
                    f"./Edge_plus_pos/pos_img{img_id}_class{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}_att{args.final_att_threshold}.jpeg")
                save_neg_map = negative_mask[i, :]
                save_neg_map[save_neg_map > 255] = 255
                # print("1280",save_edge_map.max(), save_edge_map.min(), save_edge_map.shape)
                im = Image.fromarray(np.transpose(save_neg_map, (1, 2, 0)).astype(np.uint8))
                # print("data text loc 207", f"./Edge_plus_pos/neg_img{img_id}_class{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}_att{args.final_att_threshold}.jpeg")
                im.save(
                    f"./Edge_plus_pos/neg_img{img_id}_class{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}_att{args.final_att_threshold}.jpeg")

        final_input_clip = np.concatenate((positive_mask, negative_mask), axis=0)
        # print("198 final_input_clip.shape", final_input_clip.shape) #(2 * len(gt_class_name), 3, 336, 336)
        final_input_clip = torch.from_numpy(final_input_clip).type('torch.FloatTensor')
        assert final_input_clip.shape[0] == 2 * len(gt_class_name)
        return final_input_clip, text_for_clip

    def getClassName(self, class_id, cats):
        for i in range(len(cats)):
            # print("234 get classname", cats[i]['id']==class_id, type(cats[i]['id']), type(class_id))
            if cats[i]['id'] == class_id:
                return cats[i]['name']
        return "None"

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]

        # print("46 dataset coco", img_id, type(img_id)) #  86483 <class 'int'>
        # ann_ids = coco.getAnnIds(imgIds=img_id)
        # target = coco.loadAnns(ann_ids)

        # print("50 dataset coco textloc", coco.loadImgs(img_id))
        if self.args.in_the_wild:
            img_id = torch.tensor(int(img_id))
            path = f'/home/letitiabanana/LAVIS/In_the_wild/{img_id}.jpeg'
            img0 = Image.open(path).convert('RGB')
        else:
            path = coco.loadImgs(img_id)[0]['file_name']

            img0 = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.args.search == "Unsupervised":
            # img_clip = np.transpose(self.vis_processors_clip["eval"](img0), (2, 0, 1))
            # image_input_clip = torch.from_numpy(img_clip).type('torch.FloatTensor')
            # with open(os.path.join(
            #         self.args.save_path + f"/gradcam/max_att_block_num{self.args.max_att_block_num}_del_patch_num{self.args.del_patch_num}/drop_iter0/max_att_class/",
            #         f"maxattclass_{img_id}_head{self.args.prune_att_head}.json"), 'r') as file:
            #     maxattclass_dict = json.load(file)
            # text_for_clip = maxattclass_dict[
            #     f"{img_id}_maxattcllass_top0_layer{self.args.max_att_block_num}_head{self.args.prune_att_head}"]
            image_input_clip, text_for_clip = self.Unsupervised_pn_mask_for_clipsim(self.args, img0, img_id, self.cats, self.vis_processors_clip)
            return image_input_clip, text_for_clip, img_id  # , gt_class_name  #, target, img

        if self.args.search == "Wsupervised":
            # print("245 data textloc Wsupervised")
            # img_clip = np.transpose(self.vis_processors_clip["eval"](img0), (2, 0, 1))
            # image_input_clip = torch.from_numpy(img_clip).type('torch.FloatTensor')
            # with open(os.path.join(
            #         self.args.save_path + f"/gradcam/max_att_block_num{self.args.max_att_block_num}_del_patch_num{self.args.del_patch_num}/drop_iter0/max_att_class/",
            #         f"maxattclass_{img_id}_head{self.args.prune_att_head}.json"), 'r') as file:
            #     maxattclass_dict = json.load(file)
            # text_for_clip = maxattclass_dict[
            #     f"{img_id}_maxattcllass_top0_layer{self.args.max_att_block_num}_head{self.args.prune_att_head}"]
            image_input_clip, text_for_clip = self.Wsupervised_pn_mask_for_allimgs(self.args, img0, img_id, self.vis_processors_clip, self.cats)
            # print("265 data textloc image_input_clip, text_for_clip", image_input_clip.shape, text_for_clip)
            return image_input_clip, text_for_clip, img_id  # , gt_class_name  #, target, img


        img_size = img0.size

        resized_img = img0.resize((self.args.img_size,self.args.img_size))
        norm_img = np.float32(resized_img) / 255

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.args.img_size, self.args.img_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )
        img0 = transform(img0)

        return  img_size,  norm_img, img0, path, img_id #, gt_class_name  #, target, img


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




