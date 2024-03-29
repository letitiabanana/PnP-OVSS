
import numpy as np
import os
from PIL import Image
import argparse
from os import walk
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import cv2
from matplotlib.patches import Polygon
from skimage import transform as skimage_transform
import torch
from pathlib import Path
import json
from tqdm import tqdm
from scipy.special import softmax
import math
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from torchvision import transforms


def densecrf(image, mask):
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
    image = np.ascontiguousarray(image.copy())


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

def drop_image_patch_with_highest_att(args, drop_iter, att, img_id, gt_class_name, del_patch_num):

    # For now, drop simultaneously the highest att patch for all the classes of an image
    for drop in range(drop_iter):  # range start from 0 already
        # print("255 drop iter", drop_iter)
        att_loss_path = f"{args.cam_out_dir}/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop}/highest_att_save/highest_att_{img_id}_{args.prune_att_head}.json"
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
            print(f"{args.max_att_block_num}_{img_id}", att_loss_path)
            print("217 highest att sequence of this image do not exist")
            breakpoint()

        # print("220 max_patch", drop_iter, img_id, class_name, max_patchs)
        patch_num = int(int(args.img_size) / 16)
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


def Attmap_resize(img_shape, attMap):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap = attMap/attMap.max()
    attMap = skimage_transform.resize(attMap, (img_shape), order=3, mode="constant")
    return attMap


def normalize_array(arry):
    norm_arry = (arry - arry.min()) / (arry.max() - arry.min())
    return norm_arry


def print_iou(iou, dname='voc'):
    iou_dict = {}
    for i in range(len(iou) - 1):
        iou_dict[i] = iou[i + 1]
    print(iou_dict)

    return iou_dict


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    # breakpoint()
    # print("fast hist shape check", label_true.shape, label_pred.shape, mask.shape)
    # print("fast hist shape check", label_true[mask].shape, label_pred[mask].shape)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, cats, tmp_eval_list, n_class):
    hist = np.zeros((n_class, n_class))
    # breakpoint()
    img_count = 0
    for lt, lp in zip(label_trues, label_preds):
        try:
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        except:
            print(img_count)
            print(tmp_eval_list[img_count])
            # breakpoint()
        img_count += 1
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    class_name_list = []
    for class_id in range(n_class):
        class_name_list.append(f"{class_id}_{getClassName(class_id, cats)}")
    print("69, iu.shape", iu.shape, class_name_list)
    cls_iu = dict(zip(class_name_list, iu))
    # cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def getClassName(class_id, cats):
    if class_id == 0:
        return "Background"
    try:
        return cats[class_id]
    except:
        print("Not a coco class")

        return "None class"


def flood_fill(img):
    ##### Flood fill to fill the holes ##############
    print("img sum before floodfill inv", img.sum())
    im_floodfill = cv2.bitwise_not(np.uint8(img.copy()))
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    print(" floodfill inv", im_floodfill.sum())

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # print("131", im_floodfill_inv.dtype, im_floodfill.dtype)

    # Combine the two images to get the foreground.
    im_out = cv2.bitwise_not(im_floodfill | im_floodfill_inv)
    print("img sum after floodfill", im_out.sum())
    return im_out

def normalize(att):
    if len(att.shape) > 2:
        # print("normalize att shape", att.shape)
        for i in range(att.shape[0]):
            att[i,:] = (att[i,:]-att[i,:].min())/(att[i,:].max() - att[i,:].min())
    else:
        att = (att - att.min()) / (att.max() - att.min())
    return att
#                       ###############################################
def Attmap_resize(img_shape, attMap):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img_shape), order=3, mode="constant")
    return attMap


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


def run_eval_cam(args, cats, block_num, cam_threshold, print_log=False, is_coco=False, is_coco_stuff=False):
    preds = []
    labels = []
    n_images = 0
    no_blocknum_count = 0
    # for drop_iter in range(args.drop_iter):
    #     # print("117 stats", f"{args.cam_out_dir}/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_and_img_ITM_loss_fordrop_iter{drop_iter}.json")
    #     att_loss_path = f"{args.cam_out_dir}/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_and_img_ITM_loss_fordrop_iter{drop_iter}_atthead{args.prune_att_head}.json"
    #     with open(att_loss_path, 'r') as file:
    #         globals()[f"att_loss_dict_dropiter{drop_iter}"] = json.load(file)
    COUNT = 0
    for eval_i, id in tqdm(enumerate(tmp_eval_list)):
        if not "2008_000002" in id:
            continue

        mask = np.float32(Image.open(os.path.join("/home/letitiabanana/LAVIS/mmsegmentation/data/VOCdevkit/VOC2010/SegmentationClassContext/", id + '.png')))
        # F.array(np.array(mask), cpu(0)).astype('int32')

        norm_mask = np.float32(mask)
        label = np.unique(norm_mask).tolist()
        gt_class_name = []
        for i in label:
            if i != 0:
                gt_class = cats[i]
                # print("104 gt cls", target[i]["category_id"], gt_class)
                if gt_class not in gt_class_name:
                    gt_class_name.append(gt_class)
        print(gt_class_name)
        gt = mask
        labels.append(mask)

        n_images += 1
        if args.cam_type == 'png':
            label_path = os.path.join(args.cam_out_dir, id + '.png')
            cls_labels = np.asarray(Image.open(label_path), dtype=np.uint8)
        else:
            # cam_dict = np.load(os.path.join(args.cam_out_dir, "img_" + id + ".npy"), allow_pickle=True).item()
            if args.drop_patch_eval:

                if args.drop_patch_eval == 'halving':
                    # print("111111111")

                    att_by_dropiter_list = []
                    cam_dict_drop = {}

                    for drop_iter in range(args.drop_iter):
                        if args.prune_att_head:
                            if len(args.prune_att_head) == 1:
                                cam_dict_by_dropiter = np.load(os.path.join(
                                    args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses/",
                                    f"img_{id}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy"),
                                    allow_pickle=True).item()
                                # Some classes can have no att, read as nan, incase the later operation cannot deal with nan, swith them to 0 first
                                if cam_dict_by_dropiter[args.cam_type] is not None:
                                    if np.isnan(cam_dict_by_dropiter[args.cam_type]).any():
                                        cam_dict_by_dropiter[args.cam_type][
                                            np.isnan(cam_dict_by_dropiter[args.cam_type])] = 0
                                        print("att map contains nan", id, drop_iter)

                        else:
                            cam_dict_by_dropiter = np.load(os.path.join(
                                args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses/",
                                f"img_{id}_max_blocknum_{args.max_att_block_num}.npy"),
                                allow_pickle=True).item()

                        # drop the attention of patches that have been dropped
                        if drop_iter > 0 and att_by_dropiter_list[0] is not None:
                            # print("383 sum of att before drop", cam_dict_by_dropiter[args.cam_type].sum())
                            cam_dict_by_dropiter[args.cam_type] = drop_image_patch_with_highest_att(args, drop_iter,
                                                                                                    cam_dict_by_dropiter[
                                                                                                        args.cam_type],
                                                                                                    id, gt_class_name,
                                                                                                    args.del_patch_num)
                            # cam_dict_by_dropiter[args.cam_type] = normalize(cam_dict_by_dropiter[args.cam_type])
                            # print("515 max of att after drop", cam_dict_by_dropiter[args.cam_type].max())
                            # cam_dict_by_dropiter[args.cam_type][cam_dict_by_dropiter[args.cam_type] <= 0.05] = 0  # 0.05 500 36.7
                            # print(cam_dict_by_dropiter[args.cam_type].sum())

                        att_by_dropiter_list.append(cam_dict_by_dropiter[args.cam_type])



                    if att_by_dropiter_list[0] is None:
                        # print("524 len(att_by_dropiter_list)", id, len(att_by_dropiter_list))
                        print("img has no cam", id)
                        cam_dict_drop = {f"{args.cam_type}": None, "keys": None}
                    else:


                        weighted_droppatchatt = att_by_dropiter_list[0]
                        for iter in range(args.drop_iter):  # for each drop iter
                            weighted_droppatchatt += att_by_dropiter_list[iter]

                        for cls_idx in range(weighted_droppatchatt.shape[0]):

                            weighted_droppatchatt[cls_idx][weighted_droppatchatt[cls_idx] < cam_threshold] = 0  #
                            weighted_droppatchatt[cls_idx][weighted_droppatchatt[cls_idx] >= cam_threshold] = 1  #
                        tic_post = time.perf_counter()

                        if args.postprocess:
                            if "blur" in args.postprocess:
                                print("blurring")
                                '''Apply CRF for mask refinement before threshold'''
                                cams_resize = []
                                for i in range(weighted_droppatchatt.shape[0]):
                                    att_resize = Attmap_resize(gt.shape, weighted_droppatchatt[i])

                                    att_resize = filters.gaussian_filter(att_resize, 0.05 * max(gt.shape[:2]))
                                    att_resize -= att_resize.min()
                                    att_resize /= att_resize.max()
                                    cams_resize.append(att_resize)
                                weighted_droppatchatt = np.stack(cams_resize, axis=0)
                            if "crf" in args.postprocess:
                                print("crfing")
                                background = 1 - weighted_droppatchatt.sum(axis=0)
                                background[background < 0] = 0
                                weighted_droppatchatt = np.concatenate(
                                    (np.expand_dims(background, axis=0), weighted_droppatchatt), axis=0)

                                path = "/home/letitiabanana/LAVIS/VOCdevkit/VOC2012/JPEGImages/"  # coco_thing.loadImgs(imgIds_stuff)[0]['file_name']
                                org_image = Image.open(os.path.join(path, id + '.jpg')).convert('RGB')
                                # convert_tensor = transforms.ToTensor()

                                org_image = np.asarray(org_image)
                                weighted_droppatchatt_T = torch.from_numpy(weighted_droppatchatt)
                                if len(weighted_droppatchatt_T.shape) < 3:
                                    weighted_droppatchatt_T = weighted_droppatchatt_T.unsqueeze(0)
                                map_b = densecrf(org_image, weighted_droppatchatt_T)
                                final_weighted_droppatchatt_list = []
                                for i in range(weighted_droppatchatt_T.shape[0]):
                                    if i > 0:
                                        a = map_b == i
                                        final_weighted_droppatchatt_list.append(a.astype(int))
                                weighted_droppatchatt = np.stack(final_weighted_droppatchatt_list)
                            cam_dict_drop[f"{args.cam_type}"] = weighted_droppatchatt
                            cam_dict_drop["keys"] = cam_dict_by_dropiter["keys"]
                        else:
                            cams_resize = []
                            for i in range(weighted_droppatchatt.shape[0]):
                                att_resize = Attmap_resize(gt.shape, weighted_droppatchatt[i])
                                cams_resize.append(att_resize)
                            weighted_droppatchatt = np.stack(cams_resize, axis=0)
                            cam_dict_drop[f"{args.cam_type}"] = weighted_droppatchatt
                            cam_dict_drop["keys"] = cam_dict_by_dropiter["keys"]
                        toc_post = time.perf_counter()
                        print(f"Time: Count TIME III GRADCAM {toc_post - tic_post:0.4f} seconds")

                        '''Save colored segmentation map'''
                        # from skimage import io
                        # from skimage import color
                        # from skimage import segmentation
                        # import matplotlib.pyplot as plt
                        #
                        # map_b_save = map_b.copy()
                        # # print("784", cam_dict_by_dropiter["keys"])
                        # for i, key in enumerate(cam_dict_by_dropiter["keys"]):
                        #
                        #     map_b_save[map_b == i+1] = key # map_b contains a background class
                        #
                        # print("791", np.unique(map_b_save), np.unique(gt))
                        # for i in np.unique(gt):
                        #     if i not in np.unique(map_b_save):
                        #         map_b_save[int(i),int(i)] = i
                        # print("714", np.unique(map_b_save), np.unique(gt))
                        # # assert np.unique(map_b_save) == np.unique(gt)
                        #
                        # color_map = np.random.random((59, 3))
                        # seg_map = color.label2rgb(map_b_save, org_image, bg_label=0, colors=color_map)
                        #
                        # # seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())
                        # plt.imsave(f"Qualitative_result_PSC/CRF{id}_{args.max_att_block_num}_{args.prune_att_head}_{args.postprocess}.jpeg", seg_map)
                        #
                        # gt_map = color.label2rgb(gt, org_image, bg_label=0, colors=color_map) #colors=['red','blue','green','yellow','purple','orange','brown','cyan','maroon']
                        # # gt_map = (gt_map-gt_map.min())/(gt_map.max()-gt_map.min())
                        # plt.imsave(f"Qualitative_result_PSC/GT_{id}_{args.max_att_block_num}_{args.prune_att_head}.jpeg",
                        #            gt_map)
                        ####################################






            # cam_dict = np.load(os.path.join(args.cam_out_dir, "img_" + id + f'_blocknum_{block_num}.npy'), allow_pickle=True).item()
            # print("cam dict shape", cam_dict[args.cam_type].shape[0])
            elif args.over_all_block:
                print("evaluating over all block aggregated attention....")
                att_by_block_list = []
                cam_dict_agg = {}
                for block_num in range(12):
                    cam_dict = np.load(
                        os.path.join(args.cam_out_dir, f"img_{img_id}_max_blocknum_{args.max_att_block_num}.npy"),
                        allow_pickle=True).item()

                    att_by_block_list.append(cam_dict[args.cam_type])
                stack_att = np.stack(att_by_block_list)
                # print("stack_att.shape", stack_att.shape)
                if args.over_all_block == 'avg':
                    cam_dict_agg[f"{args.cam_type}"] = np.mean(stack_att, axis=0)
                elif args.over_all_block == 'sum':
                    cam_dict_agg[f"{args.cam_type}"] = np.sum(stack_att, axis=0)
                elif args.over_all_block == 'max':
                    cam_dict_agg[f"{args.cam_type}"] = stack_att.max(axis=0)
                cam_dict_agg["keys"] = cam_dict["keys"]
                # print("131 cam dict agg", cam_dict_agg)
                # breakpoint()


            elif args.org_cam_eval:
                cam_dict = np.load(os.path.join(args.cam_out_dir,
                                                f"{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter0/img_att_forclasses/" + f"img_{img_id}_max_blocknum_{args.max_att_block_num}.npy"),
                                   allow_pickle=True).item()
            # else:
            #     no_blocknum_count += 1
            #     cam_dict = {f"{args.cam_type}": None}
            #     cam_dict_agg = {f"{args.cam_type}": None}

            if args.over_all_block:
                cam_dict = cam_dict_agg
            if args.drop_patch_eval:
                cam_dict = cam_dict_drop
            if args.drop_patch_eval_byclass:
                cam_dict = cam_dict_drop_byclass

            # print("172 cam_dict", cam_dict)
            cams_org = cam_dict[args.cam_type]

            cams_resize = []

            if cams_org is not None:
                # for i in range(cams_org.shape[0]):
                #     # breakpoint()
                #     # cams_org[i][cams_org[i] < cam_threshold] = 0
                #     # # cams_org[i][cams_org[i] > (1- cam_threshold)] = 0
                #     # cams_org[i][cams_org[i] >= cam_threshold] = 1
                #     att_resize = Attmap_resize(gt.shape, cams_org[i])
                #     class_name = getClassName(cam_dict["keys"][i], cats)
                #     # plt.imsave(f'{args.eval_mask_path}/O_img_{id}_classname_{class_name}.png', att_resize, cmap=cm.gray)
                #     if args.blur:
                #         att_resize = filters.gaussian_filter(att_resize, 0.04 * max(gt.shape[:2]))
                #         att_resize -= att_resize.min()
                #         att_resize /= att_resize.max()
                #
                #     # att_resize = flood_fill(att_resize)
                #     # plt.imsave(f'{args.eval_mask_path}/F_img_{id}_classname_{class_name}.png', att_resize, cmap=cm.gray)
                #     # att_resize[att_resize< 0.01] == 0
                #     cams_resize.append(att_resize)
                # cams = np.stack(cams_resize, axis=0)
                cams = cams_org
            else:
                cams = cams_org
                print("image", id, "has cams as ", cams)
            if 'bg' not in args.cam_type:
                if args.cam_eval_thres < 1:
                    cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
                else:
                    if cams is not None:
                        # print("445", np.unique(1 - np.max(cams, axis=0, keepdims=True))[:10])
                        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), args.cam_eval_thres)
                        cams = np.concatenate((bg_score, cams), axis=0)
                        if id == "453302":
                            print("691 check toaster", cams.shape, np.unique(cams))
                            # breakpoint()
                    else:
                        bg_score = np.power(1 - np.ones(gt.shape), args.cam_eval_thres)
                        # print("131 bg_score", bg_score.sum())
                        # breakpoint()
                        cams = bg_score

            # print("136 cams.shape", cams.shape)
            if len(cams.shape) > 2:

                keys = np.pad(cam_dict['keys'], (1, 0), mode='constant')
                cls_labels_0 = np.argmax(cams, axis=0)

                cls_labels = keys[cls_labels_0].astype(np.uint8)

            else:
                # print("387 cams.shape", cams.shape, id)
                cls_labels = cams.astype(int)
                # breakpoint()

            try:
                assert cls_labels.shape == gt.shape
                # print("391 cls_labels.shape,  gt.shape", cls_labels.shape, gt.shape, np.unique(cls_labels), np.unique(gt))
            except:
                breakpoint()

            #
            # '''Save colored segmentation map with blur only'''
            # from skimage import io
            # from skimage import color
            # from skimage import segmentation
            # import matplotlib.pyplot as plt
            # path = "/home/letitiabanana/LAVIS/VOCdevkit/VOC2012/JPEGImages/"  # coco_thing.loadImgs(imgIds_stuff)[0]['file_name']
            # org_image = Image.open(os.path.join(path, id + '.jpg')).convert('RGB')
            # org_image = np.asarray(org_image)
            # np.random.seed(0)
            # color_map = np.random.random((59, 3))
            # seg_map = color.label2rgb(cls_labels, org_image, bg_label=0, colors=color_map)
            #
            # # seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())
            # plt.imsave(f"Qualitative_result_PSC/CRF{id}_{args.max_att_block_num}_{args.prune_att_head}_{args.postprocess}.jpeg", seg_map)
            #
            # gt_map = color.label2rgb(gt, org_image, bg_label=0, colors=color_map) #colors=['red','blue','green','yellow','purple','orange','brown','cyan','maroon']
            # # gt_map = (gt_map-gt_map.min())/(gt_map.max()-gt_map.min())
            # plt.imsave(f"Qualitative_result_PSC/GT_{id}_{args.max_att_block_num}_{args.prune_att_head}_{args.postprocess}.jpeg",
            #            gt_map)
            #
            # for i,key in enumerate(np.unique(cls_labels)):
            #     class_map = cls_labels==key
            #     seg_map = color.label2rgb(class_map, org_image, bg_label=0, colors=['red','blue','green'])
            #
            #     # seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())
            #     plt.imsave(
            #         f"Qualitative_result_PSC/CRF{id}_{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}_{args.postprocess}.jpeg",
            #         seg_map)
            ###################################

        preds.append(cls_labels)

        if len(preds) != len(labels):
            print(len(preds), len(labels))
            print("pred and lables len not match", eval_i, id)
            breakpoint()

        # print("labels.shape, preds.shape", len(labels), len(preds))

    iou = scores(labels, preds, cats, tmp_eval_list, n_class=60)

    if print_log:
        print(iou)

    return iou, iou["Mean IoU"], no_blocknum_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./Eval_test_ddp_0331/img_att_forclasses/", type=str)
    parser.add_argument("--cam_type", default="att_map", type=str)
    # parser.add_argument("--split_file", default="./coco14/val.txt", type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float, help="threshold for powering the background")
    parser.add_argument("--block_num", default=6, type=int, help="from which layer you extract the attention")
    parser.add_argument("--img_root", default="/home/letitiabanana/LAVIS/mmsegmentation/data/VOCdevkit/VOC2010/", type=str)
    parser.add_argument("--save_path", default="./Pascalcontext_IOU/Pascalcontext_IOU_textloc_aftergencaption_ITM", type=str)
    parser.add_argument("--cam_threshold", default=0.2, type=float,
                        help="cam threshold that filter low attention values")
    parser.add_argument("--start_block", default=0, type=float,
                        help="start of block num range")
    parser.add_argument("--end_block", default=12, type=float,
                        help="end of block num range")
    parser.add_argument("--over_all_block", default=None,
                        help="avg/sum/max: whether calculate the avg/sum/max of attention across all blocks")
    parser.add_argument("--drop_patch_eval", default=None, type=str,
                        help="whether evaluate on drop patch attention maps")
    parser.add_argument("--drop_patch_eval_byclass", action="store_true",
                        help="whether evaluate on drop patch attention maps")
    parser.add_argument("--drop_iter", default=10, type=int,
                        help="number of drop iterations performed when obtaining attention map")
    parser.add_argument("--max_att_block_num", default=10, type=int,
                        help="number of drop iterations performed when obtaining attention map")
    parser.add_argument("--del_patch_num", default=None,
                        help="number of patch removed for each drop iterations")
    parser.add_argument("--cam_att", default="gradcam", type=str)
    parser.add_argument("--org_cam_eval", action="store_true",
                        help="whether evaluate on org attention maps")
    parser.add_argument("--blur", action="store_true",
                        help="use gaussian filter to blur the att map")
    parser.add_argument("--prune_att_head", default=None,
                        help="prune transformer attention head, indicate the retained heads")
    parser.add_argument("--eval_mask_path", default=None,
                        help="path for saving binary evaluation mask")
    parser.add_argument("--img_size", default=None,
                        help="224, 336, 448, 768")
    parser.add_argument("--postprocess", default=None,
                        help="blur, blur+crf, crf, None")
    args = parser.parse_args()


    filename_list = []
    # for (dirpath, dirnames, filenames) in walk(args.img_root):
    #     filename_list.extend(filenames)
    #     break
    # print("gt len", len(filename_list))
    # file_list = [id_.split(".")[0] for id_ in filename_list]

    with open(os.path.join("/home/letitiabanana/LAVIS/mmsegmentation/data/VOCdevkit/VOC2010/", "trainval_merged.json"), "r") as outfile:
        trainval = json.load(outfile)

    for image in trainval["images"]:
        if image["phase"] == "val":
            filename_list.append(image["file_name"])
    file_list = [id_.split(".")[0] for id_ in filename_list]
    tmp_eval_list = file_list #[:1]
    print('{} images to eval'.format(len(tmp_eval_list)))
    print("294 tmp_eval_list", tmp_eval_list)

    Path(f"./Pascalcontext_IOU").mkdir(parents=True, exist_ok=True)
    Path(f"./Drop_result").mkdir(parents=True, exist_ok=True)
    Path(f"{args.save_path}").mkdir(parents=True, exist_ok=True)
    Path(f"{args.eval_mask_path}").mkdir(parents=True, exist_ok=True)
    Path(f"./Check_crf_context").mkdir(parents=True, exist_ok=True)
    Path(f"./Qualitative_result_PSC").mkdir(parents=True, exist_ok=True)

    # no_blocknum_count = 0

    ## Check coco classname annotation

    cats = {1: 'aeroplane', 2:'bag',3 :'bed', 4:'bedclothes',
    5:'bench', 6:'bicycle', 7:'bird', 8:'boat', 9:'book', 10:'bottle',
    11:'building',12 :'bus', 13:'cabinet', 14:'car', 15:'cat', 16:'ceiling',
    17:'chair',18 :'cloth', 19:'computer', 20:'cow', 21:'cup', 22:'curtain',23 :'dog',
    24:'door', 25:'fence', 26:'floor', 27:'flower', 28:'food', 29:'grass', 30:'ground',
    31:'horse', 32:'keyboard', 33:'light', 34:'motorbike', 35:'mountain',
    36:'mouse', 37:'person', 38:'plate', 39:'platform', 40:'pottedplant', 41:'road',
    42:'rock', 43:'sheep', 44:'shelves', 45:'sidewalk',46: 'sign', 47:'sky', 48:'snow',
    49:'sofa', 50:'table', 51:'track', 52:'train', 53:'tree', 54:'truck',
    55:'tvmonitor', 56:'wall', 57:'water', 58:'window', 59:'wood'}

    nms = [i for i in cats.values()]

    # is_coco = 'coco' in args.img_root and "stuff" not in dataType_stuff
    # is_coco_stuff = 'coco' in args.img_root and "stuff" in dataType_stuff

    iou_list = []
    class_iou_dict_list = []
    no_blocknum_count_list = {}
    if args.over_all_block or args.drop_patch_eval_byclass or args.drop_patch_eval:
        start_block = 0
        end_block = 1
    else:
        start_block = args.start_block
        end_block = args.end_block

    for block_num in range(start_block, end_block):
        print("block num", block_num)

        thres_list = [1]
        # else:
        #     thres_list =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_iou = 0
        max_thres = 0
        max_class_ious = {}
        for thres in thres_list:
            args.cam_eval_thres = thres
            iou, mean_iou, no_blocknum_count = run_eval_cam(args, cats, block_num,
                                                            cam_threshold=args.cam_threshold, print_log=False)
            print(thres, iou)
            if mean_iou > max_iou:
                max_iou = mean_iou
                max_thres = thres
                max_class_ious = iou
        # args.cam_eval_thres = max_thres
        # iou, no_blocknum_count = run_eval_cam(args, coco, block_num=block_num,
        #                                       cam_threshold=args.cam_threshold, print_log=True, is_coco=is_coco)
        no_blocknum_count_list[block_num] = no_blocknum_count
        iou_list.append(max_iou)
        class_iou_dict_list.append(max_class_ious)
    print(iou_list)
    print(no_blocknum_count_list)
    with open(
            f"{args.save_path}/iou_by_blocknum_max_att_block{args.max_att_block_num}_del_patch_{args.del_patch_num}_dropiter{args.drop_iter}_camth{args.cam_threshold}.json",
            "w") as final:
        json.dump(iou_list, final)
        json.dump(class_iou_dict_list, final, indent=1)
