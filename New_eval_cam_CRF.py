
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

# from lavis.common.gradcam import getAttMap
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
            print(f"{args.max_att_block_num}_{img_id}_{class_name}", att_loss_path)
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
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
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


def getAttMap(img, attMap, blur=True, overlap=True):
    # for plotting the attention map remove the blue part
    # attMap[0] = -0.5
    # attMap = attMap - attMap.min()
    # attMap[attMap==attMap.min()] = -0.2
    if attMap.max() > 0:
        attMap = attMap /attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode="constant")
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.05 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    # attMap += 0.1
    cmap = plt.get_cmap("jet")
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = (
            1 * (1 - attMap**0.7).reshape(attMap.shape + (1,)) * img
            + (attMap**0.7).reshape(attMap.shape + (1,)) * attMapV
        )
    return attMap

def run_eval_cam(args, coco_stuff, coco_thing, cats, block_num, cam_threshold, print_log=False, is_coco=False, is_coco_stuff=False):
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
    class_count = 0
    for eval_i, id in tqdm(enumerate(tmp_eval_list)):
        # gt_file = os.path.join(args.gt_root, '%s.png' % id)
        # gt = np.array(Image.open(gt_file)).astype(np.uint8)
        # breakpoint()
        # # labels.append(gt)
        # if id not in ["314294", '443303','271471']: #["443303","145781","297830","36660"]
        #     continue

        id_coco = int(id.lstrip("0"))
        # print("66 eval cam", id_coco)
        imgIds_stuff = coco_stuff.getImgIds(imgIds=[id_coco])
        img_stuff = coco_stuff.loadImgs(imgIds_stuff)[0]
        annIds_stuff = coco_stuff.getAnnIds(imgIds=img_stuff['id'], iscrowd=None)
        anns_stuff = coco_stuff.loadAnns(annIds_stuff)



        imgIds_thing = coco_thing.getImgIds(imgIds=[id_coco])
        img_thing = coco_thing.loadImgs(imgIds_thing)[0]
        annIds_thing = coco_thing.getAnnIds(imgIds=img_thing['id'], iscrowd=None)
        anns_thing = coco_thing.loadAnns(annIds_thing)
        # print("221 eval cam", type(anns_thing))
        # breakpoint()
        if args.data_type == "cocoall":
            anns = anns_thing + anns_stuff
        elif args.data_type == "cocothing":
            anns = anns_thing
        gt_class_name = []
        for i in range(len(anns)):
            gt_class = getClassName(anns[i]["category_id"], cats)
            # print("104 gt cls", target[i]["category_id"], gt_class)
            if gt_class not in gt_class_name:
                gt_class_name.append(gt_class)
        class_count += len(gt_class_name)
        continue
        if eval_i > 4950 or eval_i < 50:
            print(gt_class_name)

        # continue
        mask_stuff = np \
            .zeros((img_stuff['height'], img_stuff['width']))
        mask_thing = np \
            .zeros((img_stuff['height'], img_stuff['width']))
        # mask_stuff += 10000
        # mask_thing += 10000
        # print("annslen", len(anns), anns)
        for i in range(len(anns_stuff)):
            # className = getClassName(anns[i]['category_id'], cats)
            pixel_value_stuff = anns_stuff[i]["category_id"]
            class_pixel_loc = np.logical_and(coco_stuff.annToMask(anns_stuff[i]), mask_stuff==0)
            mask_stuff[class_pixel_loc] = pixel_value_stuff
            # mask_stuff = np.maximum(coco_stuff.annToMask(anns_stuff[i]) * pixel_value_stuff, mask_stuff)
        for i in range(len(anns_thing)):
            # className = getClassName(anns[i]['category_id'], cats)
            pixel_value_thing = anns_thing[i]["category_id"]
            class_pixel_loc = np.logical_and(coco_thing.annToMask(anns_thing[i]), mask_thing==0)
            mask_thing[class_pixel_loc] = pixel_value_thing
            # mask_thing = np.maximum(coco_thing.annToMask(anns_thing[i]) * pixel_value_thing, mask_thing)

        if args.data_type == "cocoall":
            mask_stuff[mask_thing > 0] = mask_thing[mask_thing > 0]
            # mask_thing[mask_stuff > 0] = mask_stuff[mask_stuff > 0]
            mask = mask_stuff
        elif args.data_type == "cocothing":
            mask = mask_thing  # mask_stuff mask_thing
        # mask[mask==10000] =0
        # assert len(np.unique(mask)) == len(gt_class_name)x
        # if (mask.sum() == 0 and len(gt_class_name) > 1) or id_coco == 350679:
        #     breakpoint()
        # if "person" in gt_class_name:
        #     print("person")
        # breakpoint()
        # breakpoint()
        gt = mask
        if eval_i < 100:
            gt_save = (gt - gt.min())/(gt.max() - gt.min()) *255
            im = Image.fromarray(gt_save)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.save(f"./Check_crf_cocothing//GT_{id}_{args.max_att_block_num}_{args.prune_att_head}.jpeg")

        # print("np.unique(mask)", np.unique(mask))
        # breakpoint()
        # if gt.sum() == 0:
        #     print("no gt for image ", id)
        labels.append(mask)

        n_images += 1
        if args.cam_type == 'png':
            label_path = os.path.join(args.cam_out_dir, id + '.png')
            cls_labels = np.asarray(Image.open(label_path), dtype=np.uint8)
        else:
            # cam_dict = np.load(os.path.join(args.cam_out_dir, "img_" + id + ".npy"), allow_pickle=True).item()
            if args.drop_patch_eval_byclass:
                # print("evaluating drop patch attention....")
                att_by_dropiter_list = []
                cam_dict_drop_byclass = {}
                ITM_loss_list_bydropiter = []
                for drop_iter in range(args.drop_iter):
                    cam_dict_by_dropiter = np.load(os.path.join(
                        args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses_cancel/",
                        f"img_{id}_max_blocknum_{args.max_att_block_num}.npy"),
                        allow_pickle=True).item()

                    # cam_dict_by_dropiter[args.cam_type][cam_dict_by_dropiter[args.cam_type] < 0.1] = 0

                    class_specific_ITM_loss_list = []
                    for cls_id in cam_dict_by_dropiter["keys"]:  # save class specific weight for each drop iter

                        class_name = getClassName(cls_id, cats)
                        ITM_loss = globals()[f"att_loss_dict_dropiter{drop_iter}"][
                            f"{id}_{class_name}_ITMloss"]  # for one image, each class has its respective loss weight
                        class_specific_ITM_loss_list.append(ITM_loss)

                    ITM_loss_list_bydropiter.append(class_specific_ITM_loss_list)

                    att_by_dropiter_list.append(cam_dict_by_dropiter[args.cam_type])
                weight_from_byclassITMloss = np.stack(ITM_loss_list_bydropiter)
                # print("weight_from_byclassITMloss.shape", weight_from_byclassITMloss.shape)
                for i in range(weight_from_byclassITMloss.shape[1]):
                    weight_from_byclassITMloss[:, i] = list(softmax([y - x for x, y in
                                                                     zip(weight_from_byclassITMloss[:, i][:-1],
                                                                         weight_from_byclassITMloss[:, i][1:])])) + [0.1]
                    print("162 weight_from_byclassITMloss[:,i]", weight_from_byclassITMloss[:, i])
                # stack_att = np.stack(att_by_dropiter_list)
                # print("159 weight_from_byclassITMloss.shape, stack_att.shape", weight_from_byclassITMloss.shape, att_by_dropiter_list[0].shape, att_by_dropiter_list[0].sum())
                # breakpoint()

                # ITM_loss_list = [0.5] + softmax([y - x for x, y in zip(ITM_loss_list[:-1], ITM_loss_list[1:])])
                weighted_droppatchatt = np.zeros(att_by_dropiter_list[0].shape)
                for iter in range(weight_from_byclassITMloss.shape[0]):

                    for cls_idx in range(weight_from_byclassITMloss.shape[1]):
                        weighted_droppatchatt[cls_idx, :, :] += att_by_dropiter_list[iter][cls_idx, :, :] * \
                                                                weight_from_byclassITMloss[iter, cls_idx]

                print("177 weighted_droppatchatt.shape", weighted_droppatchatt.shape, weighted_droppatchatt.sum(),
                      att_by_dropiter_list[iter][cls_idx, :, :].shape, weight_from_byclassITMloss[iter, cls_idx])

                cam_dict_drop_byclass[f"{args.cam_type}"] = weighted_droppatchatt
                cam_dict_drop_byclass["keys"] = cam_dict_by_dropiter["keys"]

            elif args.drop_patch_eval:

                if args.drop_patch_eval == 'ITMloss':

                    att_by_dropiter_list = []
                    cam_dict_drop = {}
                    ITM_loss_list = []
                    for drop_iter in range(args.drop_iter):
                        if args.prune_att_head:
                            if len(args.prune_att_head) == 1:
                                cam_dict_by_dropiter = np.load(os.path.join(
                                    args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses_cancel/",
                                    f"img_{id}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy"),
                                    allow_pickle=True).item()
                                if cam_dict_by_dropiter[args.cam_type] is not None:
                                    if np.isnan(cam_dict_by_dropiter[args.cam_type]).any():
                                        cam_dict_by_dropiter[args.cam_type][
                                            np.isnan(cam_dict_by_dropiter[args.cam_type])] = 0
                                        print("att map contains nan", id, drop_iter)
                                # if cam_dict_by_dropiter[args.cam_type] is not None:
                                #     for cls_id, cls_name in enumerate(gt_class_name):
                                #         # print("267 cam dict check normalization",id, cls_name, cam_dict_by_dropiter[args.cam_type][cls_id,:].max(),cam_dict_by_dropiter[args.cam_type][cls_id,:].min())
                                #         cam_dict_by_dropiter[args.cam_type][cls_id,:] = (cam_dict_by_dropiter[args.cam_type][cls_id,:] - cam_dict_by_dropiter[args.cam_type][cls_id,:].min()) / (cam_dict_by_dropiter[args.cam_type][cls_id,:].max() - cam_dict_by_dropiter[args.cam_type][cls_id,:].min())
                            elif args.prune_att_head == "369":
                                cam_dict_by_atthead_list = []
                                for att_head in [3, 6, 9]:
                                    cam_dict_by_dropiter = np.load(os.path.join(
                                        args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses_cancel/",
                                        f"img_{id}_max_blocknum_{args.max_att_block_num}_atthead_{att_head}.npy"),
                                        allow_pickle=True).item()
                                    cam_dict_by_atthead_list.append(cam_dict_by_dropiter[args.cam_type])
                                if cam_dict_by_atthead_list[0] is not None:
                                    cam_dict_by_dropiter[args.cam_type] = cam_dict_by_atthead_list[0] - 0.3 * \
                                                                          cam_dict_by_atthead_list[1] + \
                                                                          cam_dict_by_atthead_list[2]

                        else:
                            cam_dict_by_dropiter = np.load(os.path.join(
                                args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses_cancel/",
                                f"img_{id}_max_blocknum_{args.max_att_block_num}.npy"),
                                allow_pickle=True).item()

                        if drop_iter > 0 and att_by_dropiter_list[0] is not None:
                            # print("243 sum of att before drop", cam_dict_by_dropiter[args.cam_type].sum())
                            tic1 = time.perf_counter()
                            cam_dict_by_dropiter[args.cam_type] = drop_image_patch_with_highest_att(args, drop_iter,
                                                                                                    cam_dict_by_dropiter[
                                                                                                        args.cam_type],
                                                                                                    id, gt_class_name,
                                                                                                    args.del_patch_num)
                            toc1 = time.perf_counter()
                            print(f"Time: save image by class att and union image use {toc1 - tic1:0.4f} seconds")
                            # print("243 sum of att after drop", cam_dict_by_dropiter[args.cam_type].sum())

                        att_by_dropiter_list.append(cam_dict_by_dropiter[args.cam_type])
                        # cam_dict_by_dropiter[args.cam_type][cam_dict_by_dropiter[args.cam_type] < 0.1*(args.drop_iter-drop_iter)] = 0

                        ITM_loss = globals()[f"att_loss_dict_dropiter{drop_iter}"][
                            f"{id}_ITMloss"]  # for one image, each class has its respective loss weight

                        ITM_loss_list.append(ITM_loss)

                    if att_by_dropiter_list[0] is None:
                        print("img has no cam", id)
                        cam_dict_drop = {f"{args.cam_type}": None, "keys": None}
                    else:
                        if args.drop_iter > 1:
                            # stack_att = np.stack(att_by_dropiter_list)
                            # ITM_loss_list = [1/args.drop_iter] * args.drop_iter #softmax([y - x for x, y in zip(ITM_loss_list[:-1], ITM_loss_list[1:])]) + [0.5]
                            ITM_loss_list = [0.5] + np.concatenate(
                                ([0.5], softmax([y - x for x, y in zip(ITM_loss_list[:-1], ITM_loss_list[1:])])))  # +
                        else:
                            ITM_loss_list = [1]
                            # print("ITM_loss_list", ITM_loss_list)

                        weighted_droppatchatt_byITMloss = np.zeros(att_by_dropiter_list[0].shape)
                        for ind, w in enumerate(ITM_loss_list):
                            weighted_droppatchatt_byITMloss += att_by_dropiter_list[ind] * w

                        # try:
                        #     assert np.array_equal(weighted_droppatchatt_byITMloss,att_by_dropiter_list[0])
                        # except:
                        #     breakpoint()
                        weighted_droppatchatt_byITMloss[weighted_droppatchatt_byITMloss < cam_threshold] = 0
                        weighted_droppatchatt_byITMloss[weighted_droppatchatt_byITMloss >= cam_threshold] = 1

                        # print("271 unique weighted_droppatchatt_byITMloss", np.unique(weighted_droppatchatt_byITMloss))
                        # # cams_org[i][cams_org[i] > (1- cam_threshold)] = 0
                        # cams_org[i][cams_org[i] >= cam_threshold] = 1

                        cam_dict_drop[f"{args.cam_type}"] = weighted_droppatchatt_byITMloss
                        cam_dict_drop["keys"] = cam_dict_by_dropiter["keys"]
                elif args.drop_patch_eval == 'clipsim':
                    # print("111111111")

                    att_by_dropiter_list = []
                    cam_dict_drop = {}
                    clipsim_list_bydropiter = []
                    for drop_iter in range(args.drop_iter):
                        # cam_dict_by_dropiter = np.load(os.path.join(
                        #     args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses/",
                        #     f"img_{id}_max_blocknum_{args.max_att_block_num}.npy"),
                        #     allow_pickle=True).item()
                        # # print("cam_dict_by_dropiter max min",cam_dict_by_dropiter[args.cam_type].max(), cam_dict_by_dropiter[args.cam_type].min(), type(cam_dict_by_dropiter[args.cam_type][0,0,0]))
                        # # if cam_dict_by_dropiter[args.cam_type] is not None:
                        # #     cam_dict_by_dropiter[args.cam_type][cam_dict_by_dropiter[args.cam_type] < 0.05] = 0
                        if args.prune_att_head:
                            if len(args.prune_att_head) == 1:
                                cam_dict_by_dropiter = np.load(os.path.join(
                                    args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses_cancel/",
                                    f"img_{id}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy"),
                                    allow_pickle=True).item()
                                # Some classes can have no att, read as nan, incase the later operation cannot deal with nan, swith them to 0 first
                                if cam_dict_by_dropiter[args.cam_type] is not None:
                                    if np.isnan(cam_dict_by_dropiter[args.cam_type]).any():
                                        cam_dict_by_dropiter[args.cam_type][np.isnan(cam_dict_by_dropiter[args.cam_type])] = 0
                                        print("att map contains nan", id, drop_iter)
                                # if cam_dict_by_dropiter[args.cam_type] is not None:
                                #     for cls_id, cls_name in enumerate(gt_class_name):
                                #         # print("267 cam dict check normalization",id, cls_name, cam_dict_by_dropiter[args.cam_type][cls_id,:].max(),cam_dict_by_dropiter[args.cam_type][cls_id,:].min())
                                #         cam_dict_by_dropiter[args.cam_type][cls_id,:] = (cam_dict_by_dropiter[args.cam_type][cls_id,:] - cam_dict_by_dropiter[args.cam_type][cls_id,:].min()) / (cam_dict_by_dropiter[args.cam_type][cls_id,:].max() - cam_dict_by_dropiter[args.cam_type][cls_id,:].min())
                            elif args.prune_att_head == "369":
                                cam_dict_by_atthead_list = []
                                for att_head in [3, 6, 9]:
                                    cam_dict_by_dropiter = np.load(os.path.join(
                                        args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses_cancel/",
                                        f"img_{id}_max_blocknum_{args.max_att_block_num}_atthead_{att_head}.npy"),
                                        allow_pickle=True).item()
                                    cam_dict_by_atthead_list.append(cam_dict_by_dropiter[args.cam_type])
                                if cam_dict_by_atthead_list[0] is not None:
                                    cam_dict_by_dropiter[args.cam_type] = cam_dict_by_atthead_list[0] - 0.3 * \
                                                                          cam_dict_by_atthead_list[1] + \
                                                                          cam_dict_by_atthead_list[2]

                        else:
                            cam_dict_by_dropiter = np.load(os.path.join(
                                args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses_cancel/",
                                f"img_{id}_max_blocknum_{args.max_att_block_num}.npy"),
                                allow_pickle=True).item()

                        # drop the attention of patches that have been dropped
                        if drop_iter > 0 and att_by_dropiter_list[0] is not None:
                            # print("383 sum of att before drop", cam_dict_by_dropiter[args.cam_type].sum())
                            cam_dict_by_dropiter[args.cam_type] = drop_image_patch_with_highest_att(args, drop_iter,
                                                                                                    cam_dict_by_dropiter[args.cam_type],
                                                                                                    id, gt_class_name,
                                                                                                    args.del_patch_num)
                            cam_dict_by_dropiter[args.cam_type] = normalize(cam_dict_by_dropiter[args.cam_type])
                            # print("515 max of att after drop", cam_dict_by_dropiter[args.cam_type].max())
                            cam_dict_by_dropiter[args.cam_type][cam_dict_by_dropiter[args.cam_type] <= 0.05] = 0 #0.05 500 36.7
                            # print(cam_dict_by_dropiter[args.cam_type].sum())

                        att_by_dropiter_list.append(cam_dict_by_dropiter[args.cam_type])
                        class_specific_clipsim_list = []

                        for cls_id in cam_dict_by_dropiter["keys"]:  # save class specific weight for each drop iter
                            class_name = getClassName(cls_id, cats)
                            try:
                                clip_sim_byclass = globals()[f"att_loss_dict_dropiter{drop_iter}"][f"{id}_classname{class_name}_clipsim"]
                                class_specific_clipsim_list.append(clip_sim_byclass)
                            except:
                                clip_sim_byclass = 0
                                class_specific_clipsim_list.append(clip_sim_byclass)
                                print("576 no clipsim", drop_iter,f"{id}_classname{class_name}_clipsim", COUNT)
                                COUNT +=1
                        clipsim_list_bydropiter.append(class_specific_clipsim_list)

                        #     continue
                    try:
                        assert len(clipsim_list_bydropiter) == args.drop_iter
                    except:
                        breakpoint()
                    if att_by_dropiter_list[0] is None:
                        # print("524 len(att_by_dropiter_list)", id, len(att_by_dropiter_list))
                        print("img has no cam", id)
                        cam_dict_drop = {f"{args.cam_type}": None, "keys": None}
                    else:
                        byclass_clipsim = np.stack(clipsim_list_bydropiter)
                        # print("395 byclass_clipsim", id, len(gt_class_name), byclass_clipsim.shape)
                        weight_from_byclass_clipsim = np.ones(byclass_clipsim.shape)

                        for j in range(byclass_clipsim.shape[1]):  # for each class
                            # print("238 diff", class_name, diff)
                            for i in range(1, byclass_clipsim.shape[0]):  # for each drop iter

                                if byclass_clipsim[i, j] < 0.25:
                                    weight_from_byclass_clipsim[i:, j] = 0  #clip sim < 0.25 everything behind deleted
                                    break
                                    # weight_from_byclass_clipsim[i, j] = 0 #clip sim > 0.25 add bad


                        weighted_droppatchatt = np.zeros(att_by_dropiter_list[0].shape)
                        # print("450 att_by_dropiter_list[0].shape", att_by_dropiter_list[0].shape)
                        for cls_idx in range(weight_from_byclass_clipsim.shape[1]):
                            for iter in range(weight_from_byclass_clipsim.shape[0]):
                                weighted_droppatchatt[cls_idx, :, :] += att_by_dropiter_list[iter][cls_idx, :, :] * \
                                                                        weight_from_byclass_clipsim[iter, cls_idx]

                        for cls_idx in range(weight_from_byclass_clipsim.shape[1]):
                            weighted_droppatchatt[cls_idx][weighted_droppatchatt[cls_idx] < cam_threshold] = 0  #
                            weighted_droppatchatt[cls_idx][weighted_droppatchatt[cls_idx] >= cam_threshold] = 1  #
                        if id == "453302":
                            print("check toaster", gt_class_name, weight_from_byclass_clipsim)
                            # breakpoint()

                        ######################################


                        cam_dict_drop[f"{args.cam_type}"] = weighted_droppatchatt
                        cam_dict_drop["keys"] = cam_dict_by_dropiter["keys"]
                elif args.drop_patch_eval == 'halving':
                    # print("111111111")

                    att_by_dropiter_list = []
                    cam_dict_drop = {}

                    for drop_iter in range(args.drop_iter):
                        if args.prune_att_head:
                            if len(args.prune_att_head) == 1 or args.prune_att_head == "10" or args.prune_att_head == "11" or args.prune_att_head == "average":
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

                        tic_combinedrop = time.perf_counter()

                        # drop the attention of patches that have been dropped
                        if drop_iter > 0 and att_by_dropiter_list[0] is not None:
                            # print("383 sum of att before drop", cam_dict_by_dropiter[args.cam_type].sum())
                            cam_dict_by_dropiter[args.cam_type] = drop_image_patch_with_highest_att(args, drop_iter,
                                                                                                    cam_dict_by_dropiter[args.cam_type],
                                                                                                    id, gt_class_name,
                                                                                                    args.del_patch_num)
                            # cam_dict_by_dropiter[args.cam_type] = normalize(cam_dict_by_dropiter[args.cam_type])
                            # print("515 max of att after drop", cam_dict_by_dropiter[args.cam_type].max())
                            # cam_dict_by_dropiter[args.cam_type][cam_dict_by_dropiter[args.cam_type] <= 0.05] = 0  # 0.05 500 36.7
                            # print(cam_dict_by_dropiter[args.cam_type].sum())
                        toc_combinedrop = time.perf_counter()
                        print(
                            f"Count 3 {id} Time: Combine drop during eval {toc_combinedrop - tic_combinedrop:0.4f} seconds")
                        att_by_dropiter_list.append(cam_dict_by_dropiter[args.cam_type])
                        class_specific_clipsim_list = []

                        # for cls_id in cam_dict_by_dropiter["keys"]:  # save class specific weight for each drop iter
                        #     class_name = getClassName(cls_id, cats)
                        #     try:
                        #         clip_sim_byclass = globals()[f"att_loss_dict_dropiter{drop_iter}"][
                        #             f"{id}_classname{class_name}_clipsim"]
                        #         class_specific_clipsim_list.append(clip_sim_byclass)
                        #     except:
                        #         clip_sim_byclass = 0
                        #         class_specific_clipsim_list.append(clip_sim_byclass)
                        #         print("576 no clipsim", drop_iter, f"{id}_classname{class_name}_clipsim", COUNT)
                        #         COUNT += 1
                        # clipsim_list_bydropiter.append(class_specific_clipsim_list)

                        #     continue
                    # try:
                    #     assert len(clipsim_list_bydropiter) == args.drop_iter
                    # except:
                    #     breakpoint()
                    if att_by_dropiter_list[0] is None:
                        # print("524 len(att_by_dropiter_list)", id, len(att_by_dropiter_list))
                        print("img has no cam", id)
                        cam_dict_drop = {f"{args.cam_type}": None, "keys": None}
                    else:
                        # byclass_clipsim = np.stack(clipsim_list_bydropiter)
                        # print("395 byclass_clipsim", id, len(gt_class_name), byclass_clipsim.shape)
                        # weight_from_byclass_clipsim = np.ones((args.drop_iter, len(cam_dict_by_dropiter["keys"])))
                        #
                        # for j in range(byclass_clipsim.shape[1]):  # for each class
                        #     # print("238 diff", class_name, diff)
                        #     for i in range(1, byclass_clipsim.shape[0]):  # for each drop iter
                        #
                        #         if byclass_clipsim[i, j] < 0.25:
                        #             weight_from_byclass_clipsim[i:, j] = 0  # clip sim < 0.25 everything behind deleted
                        #             break
                        #             # weight_from_byclass_clipsim[i, j] = 0 #clip sim > 0.25 add bad

                        # weighted_droppatchatt = np.zeros(att_by_dropiter_list[0].shape)
                        # # print("450 att_by_dropiter_list[0].shape", att_by_dropiter_list[0].shape)
                        # for cls_idx in range(weight_from_byclass_clipsim.shape[1]):# for each class
                        #     for iter in range(weight_from_byclass_clipsim.shape[0]):# for each drop iter
                        #         weighted_droppatchatt[cls_idx, :, :] += att_by_dropiter_list[iter][cls_idx, :, :]


                        weighted_droppatchatt = att_by_dropiter_list[0]
                        for iter in range(args.drop_iter):  # for each drop iter
                            weighted_droppatchatt += att_by_dropiter_list[iter]

                        #
                        # '''Plot salience drop attention map'''
                        # path = coco_thing.loadImgs(imgIds_stuff)[0]['file_name']
                        # org_image = Image.open(os.path.join(args.img_root, path)).convert('RGB')
                        # # convert_tensor = transforms.ToTensor()
                        # norm_img = np.float32(org_image) / 255
                        #
                        # for i in range(weighted_droppatchatt.shape[0]):
                        #     class_map = weighted_droppatchatt[i]
                        #     # class_map[class_map>0.15] = 0.9
                        #     gradcam_image_c = getAttMap(norm_img, class_map, blur=True)  # range between 0-1
                        #     im_gradcam_c = Image.fromarray((gradcam_image_c * 255).astype(np.uint8), 'RGB')
                        #     im_gradcam_c.save(
                        #         f"Qualitative_result_{args.data_type}/Pred_att_{id}_{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}_{args.drop_iter}_{args.cam_att}.jpeg")

                        '''Thresholding the attention map'''
                        #
                        for cls_idx in range(weighted_droppatchatt.shape[0]):
                            weighted_droppatchatt[cls_idx][weighted_droppatchatt[cls_idx] < 0.15] = 0  #
                            weighted_droppatchatt[cls_idx][weighted_droppatchatt[cls_idx] >= 0.15] = 1  #

                        '''Apply CRF for mask refinement before threshold'''
                        if args.postprocess:
                            tic_postprocess = time.perf_counter()

                            if "blur" in args.postprocess:
                                print("blurring")
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
                                if args.data_type == "cocoall":
                                # cocoall already include annotation for almost everything in the picture, background are actully foreground under this setting
                                    pass
                                else:
                                    background = 1 - weighted_droppatchatt.sum(axis=0)
                                    background[background < 0] = 0
                                    weighted_droppatchatt = np.concatenate((np.expand_dims(background, axis=0), weighted_droppatchatt), axis=0)


                                path = coco_thing.loadImgs(imgIds_stuff)[0]['file_name']
                                org_image = Image.open(os.path.join(args.img_root, path)).convert('RGB')
                                # convert_tensor = transforms.ToTensor()
                                org_image = np.asarray(org_image)
                                weighted_droppatchatt_T = torch.from_numpy(weighted_droppatchatt)
                                if len(weighted_droppatchatt_T.shape) < 3:
                                    weighted_droppatchatt_T = weighted_droppatchatt_T.unsqueeze(0)
                                map_b = densecrf(org_image, weighted_droppatchatt_T)
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
                            cam_dict_drop[f"{args.cam_type}"] = weighted_droppatchatt
                            cam_dict_drop["keys"] = cam_dict_by_dropiter["keys"]

                            toc_postprocess = time.perf_counter()
                            print(
                                f"Count 4 {id} Time: Post processing during eval {toc_postprocess - tic_postprocess:0.4f} seconds")
                        else:
                            cams_resize = []
                            for i in range(weighted_droppatchatt.shape[0]):
                                att_resize = Attmap_resize(gt.shape, weighted_droppatchatt[i])
                                cams_resize.append(att_resize)
                            weighted_droppatchatt = np.stack(cams_resize, axis=0)
                            cam_dict_drop[f"{args.cam_type}"] = weighted_droppatchatt
                            cam_dict_drop["keys"] = cam_dict_by_dropiter["keys"]



                        # '''Save colored segmentation map'''
                        # from skimage import io
                        # from skimage import color
                        # from skimage import segmentation
                        # import matplotlib.pyplot as plt
                        # 
                        # map_b_save = map_b.copy()
                        # # print("784", cam_dict_by_dropiter["keys"])
                        # for i, key in enumerate(cam_dict_by_dropiter["keys"]):
                        # 
                        #     map_b_save[map_b == i] = key

                        # print("791", np.unique(map_b_save),np.unique(gt) )

                        # for i in np.unique(gt):
                        #     if i not in np.unique(map_b_save):
                        #         map_b_save[int(i)] = i
                        # print("714", np.unique(map_b_save), np.unique(gt))
                        # assert np.unique(map_b_save) == np.unique(gt)

                        # color_map = np.random.shuffle(np.random.random((183, 3)))
                        # seg_map = color.label2rgb(map_b_save, org_image, bg_label=0, colors=color_map)
                        # 
                        # # seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())
                        # plt.imsave(f"Qualitative_result_{args.data_type}/CRF{id}_{args.max_att_block_num}_{args.prune_att_head}.jpeg", seg_map)
                        # 
                        # gt_map = color.label2rgb(gt, org_image, bg_label=0, colors=color_map) #colors=['red','blue','green','yellow','purple','orange','brown','cyan','maroon']
                        # # gt_map = (gt_map-gt_map.min())/(gt_map.max()-gt_map.min())
                        # plt.imsave(f"Qualitative_result_{args.data_type}/GT_{id}_{args.max_att_block_num}_{args.prune_att_head}.jpeg",
                        #            gt_map)


                        # if len(np.unique(map_b_save)) > 1:
                        #     print("795", np.unique(map_b_save), np.unique(gt)[1:])
                        #     assert (np.unique(map_b_save) == np.unique(gt)[1:]).all()
                        # else:
                        #     assert np.unique(map_b_save) == np.unique(gt)[1:]



                        # if eval_i < 100:
                        #     map_b = map_b/(map_b.max()-map_b.min()) * 255
                        #     im = Image.fromarray(map_b)
                        #     if im.mode != 'RGB':
                        #         im = im.convert('RGB')
                        #     im.save(f"./Check_crf_cocothing/With_threshold_CRF{id}_{args.max_att_block_num}_{args.prune_att_head}.jpeg")

                        # #####################################
                        '''Apply CRF for mask refinement after threshold '''
                        # path = coco_thing.loadImgs(imgIds_stuff)[0]['file_name']
                        # org_image = Image.open(os.path.join(args.img_root, path)).convert('RGB')
                        # # convert_tensor = transforms.ToTensor()
                        # org_image = np.asarray(org_image)
                        # weighted_droppatchatt_T = torch.from_numpy(weighted_droppatchatt)
                        # map_af = densecrf(org_image, weighted_droppatchatt_T)
                        # final_weighted_droppatchatt_list = []
                        # for i in range(weighted_droppatchatt.shape[0]):
                        #     a = map_af == i
                        #     final_weighted_droppatchatt_list.append(a.astype(int))
                        # final_weighted_droppatchatt = np.stack(final_weighted_droppatchatt_list)
                        # print("735", id_coco, final_weighted_droppatchatt.shape)
                        #
                        # map_af = map_af/(map_af.max()-map_af.min()) * 255
                        # im = Image.fromarray(map_af)
                        # if im.mode != 'RGB':
                        #     im = im.convert('RGB')
                        # im.save(
                        #     f"./Check_crf_cocothing//After_threshold_CRF{id}_{args.max_att_block_num}_{args.prune_att_head}_attthresh{args.cam_threshold}.jpeg")
                        #


                        cam_dict_drop[f"{args.cam_type}"] = weighted_droppatchatt
                        cam_dict_drop["keys"] = cam_dict_by_dropiter["keys"]


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

            # try:
            #     cam_dict = np.load(os.path.join(args.cam_out_dir, "img_" + id + f'_blocknum_11.npy'),
            #                        allow_pickle=True).item()
            # except:
            #     no_blocknum_count += 1
            #     cam_dict = {f"{args.cam_type}": None}
            #
            # if cam_dict[f"{args.cam_type}"] is not None:
            #     cams_org = cam_dict[f"{args.cam_type}"][block_num:(block_num + 2), :, :]
            #     # print("118 cams_org.shape", cams_org.shape)
            #     # cam_dict = cam_dict[block_num:(block_num + 2), :, :]
            # else:
            #     cams_org = cam_dict[args.cam_type]

            # if cams_org is not None:
            #     print("105 cams_org shape", cams_org.shape, cams_org.min(), cams_org.max(), np.unique(cams_org))
            # else:
            #     print("cams_org is none")

            # print("103 gt shape", gt.shape)
            cams_resize = []

            if cams_org is not None:
                '''skip blurring and resizing when applying CRF'''
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
                # cls_labels_save = cls_labels / (cls_labels.max() - cls_labels.min()) * 255
                # im = Image.fromarray(cls_labels_save)
                # if im.mode != 'RGB':
                #     im = im.convert('RGB')
                # im.save(
                #     f"./Check_crf_cocothing//Blursave_CRF{id}_{args.max_att_block_num}_{args.prune_att_head}_attthresh{args.cam_threshold}.jpeg")

            else:
                # print("387 cams.shape", cams.shape, id)
                cls_labels = cams.astype(int)

                # breakpoint()
            # print("140 cls_labels_0.shape", cls_labels_0.shape)
            # breakpoint()

            try:
                assert cls_labels.shape == gt.shape
                # print("391 cls_labels.shape,  gt.shape", cls_labels.shape, gt.shape, np.unique(cls_labels), np.unique(gt))
            except:
                breakpoint()


            '''Save colored segmentation map'''

            path = coco_thing.loadImgs(imgIds_stuff)[0]['file_name']
            org_image = Image.open(os.path.join(args.img_root, path)).convert('RGB')
            org_image = np.asarray(org_image)
            from skimage import io
            from skimage import color
            from skimage import segmentation
            import matplotlib.pyplot as plt

            np.random.seed(0)
            color_map = np.random.shuffle(np.random.random((183, 3)))
            seg_map = color.label2rgb(cls_labels, org_image, bg_label=0, colors=color_map)

            # seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())
            plt.imsave(f"Qualitative_result_{args.data_type}/CRF{id}_{args.max_att_block_num}_{args.prune_att_head}.jpeg", seg_map)

            gt_map = color.label2rgb(gt, org_image, bg_label=0, colors=color_map) #colors=['red','blue','green','yellow','purple','orange','brown','cyan','maroon']
            # gt_map = (gt_map-gt_map.min())/(gt_map.max()-gt_map.min())
            plt.imsave(f"Qualitative_result_{args.data_type}/GT_{id}_{args.max_att_block_num}_{args.prune_att_head}.jpeg",
                       gt_map)

            for i,key in enumerate(np.unique(cls_labels)):
                class_map = cls_labels==key
                seg_map = color.label2rgb(class_map, org_image, bg_label=0, colors=color_map)

                # seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())
                plt.imsave(
                    f"Qualitative_result_{args.data_type}/CRF{id}_{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}.jpeg",
                    seg_map)

                gt_class_map = gt==key
                gt_map = color.label2rgb(gt_class_map, org_image, bg_label=0, colors=color_map)

                # seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())
                plt.imsave(
                    f"Qualitative_result_{args.data_type}/GT_{id}_{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}.jpeg",
                    gt_map)
                # class_map = class_map.astype(int)
                gradcam_image_c = getAttMap(org_image, class_map, blur=False)  # range between 0-1
                im_gradcam_c = Image.fromarray((gradcam_image_c * 255).astype(np.uint8), 'RGB')
                im_gradcam_c.save(f"Qualitative_result_{args.data_type}/Pred_att_{id}_{gt_class_name[i]}_{args.max_att_block_num}_{args.prune_att_head}.jpeg")

        preds.append(cls_labels)

        if len(preds) != len(labels):
            print(len(preds), len(labels))
            print("pred and lables len not match", eval_i, id)
            breakpoint()



        # print("labels.shape, preds.shape", len(labels), len(preds))
    if is_coco_stuff:
        iou = scores(labels, preds, cats, tmp_eval_list, n_class=184)
    else:
        iou = scores(labels, preds, cats, tmp_eval_list, n_class=21 if not is_coco else 91)

    if print_log:
        print(iou)
    print("total number of masks in cocoall", class_count)
    return iou, iou["Mean IoU"], no_blocknum_count

    # def softmax(logits):
    #     bottom = sum([math.exp(x) for x in logits])
    #     softmax = [math.exp(x)/bottom for x in logits]
    #     return softmax
    #
    # def softmax(x):
    #     """Compute softmax values for each sets of scores in x."""
    #     e_x = np.exp(x - np.max(x))
    #     return e_x / e_x.sum(axis=0) # only difference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./Eval_test_ddp_0331/img_att_forclasses/", type=str)
    parser.add_argument("--cam_type", default="att_map", type=str)
    # parser.add_argument("--split_file", default="./coco14/val.txt", type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float, help="threshold for powering the background")
    parser.add_argument("--block_num", default=6, type=int, help="from which layer you extract the attention")
    parser.add_argument("--img_root", default="./coco/images/val2017/", type=str)
    parser.add_argument("--save_path", default="./COCO_IOU/COCO_IOU_textloc_aftergencaption_ITM", type=str)
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
    parser.add_argument("--data_type", default=None,
                        help="cocothing or cocoall")
    parser.add_argument("--img_size", default=None,
                        help="224, 336, 448, 768")
    parser.add_argument("--postprocess", default=None,
                        help="blur, blur+crf, crf, None")
    args = parser.parse_args()

    filename_list = []
    for (dirpath, dirnames, filenames) in walk(args.img_root):
        filename_list.extend(filenames)
        break
    print("gt len", len(filename_list))
    file_list = [id_.lstrip("0").rstrip().split(".") for id_ in filename_list]
    tmp_eval_list = [x[0] for x in file_list]
    # tmp_eval_list = tmp_eval_list[:100]
    print('{} images to eval'.format(len(tmp_eval_list)))
    print("294 tmp_eval_list", tmp_eval_list)
    # print('{} tmp images to eval'.format(len(tmp_eval_list)))

    # filename_list = []
    # for (dirpath, dirnames, filenames) in walk(args.cam_out_dir + f"/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter9/img_att_forclasses/"):
    #     filename_list.extend(filenames)
    #     break
    # print("cam len", filename_list)
    # tmp_eval_list = []
    # for x in filename_list:
    #     if x.split("_")[1] not in tmp_eval_list:
    #         tmp_eval_list.append(x.split("_")[1])

    #
    # word_size = 2
    # # if not os.path.exists(f"{args.cam_out_dir}/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{args.drop_iter - 1}/highest_att_and_img_ITM_loss_fordrop_iter{args.drop_iter - 1}_atthead{args.prune_att_head}.json"):
    # for drop_iter in range(4): #args.drop_iter
    #     print(drop_iter)
    #     att_all = {}
    #     for rank in range(word_size):
    #         print(rank)
    #         with open(
    #                 f"{args.cam_out_dir}/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_and_img_ITM_loss_fordrop_iter{drop_iter}_atthead{args.prune_att_head}_{rank}.json",
    #                 "r") as outfile:
    #             att_before = json.load(outfile)
    #         att_all.update(att_before)
    #     with open(
    #             f"{args.cam_out_dir}/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_and_img_ITM_loss_fordrop_iter{drop_iter}.json",
    #             "w") as outfile:
    #         json.dump(att_all, outfile, cls=NumpyArrayEncoder)

    # att_loss_path = f"{args.cam_out_dir}/{args.cam_att}/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter0/highest_att_and_img_ITM_loss_fordrop_iter0_atthead{args.prune_att_head}.json"  # _atthead{args.prune_att_head}
    #
    # # att_loss_path = "./Cbatch_Eval_test_ddp_0610_combinelabelascaption_softmaxloss_withitercheck/gradcam/max_att_block_num8_del_patch_num7/drop_iter9/highest_att_and_img_ITM_loss_fordrop_iter9.json"
    # with open(att_loss_path, 'r') as file:
    #     att_loss_dict = json.load(file)
    # tmp_eval_list = []
    # for x in att_loss_dict.keys():
    #     if "ITMloss" in x:
    #         if x.split("_")[0] not in tmp_eval_list:
    #             tmp_eval_list.append(x.split("_")[0])
    #
    # # tmp_eval_list = tmp_eval_list[]
    # print("294 tmp_eval_list", tmp_eval_list)
    # print('{} tmp images to eval'.format(len(tmp_eval_list)))

    # if 'voc' in args.cam_out_dir:
    #     eval_list = list(np.loadtxt(args.split_file, dtype=str))
    # elif 'coco' in args.cam_out_dir:
    #     file_list = tuple(open(args.split_file, "r"))
    #     file_list = [id_.rstrip().split(" ") for id_ in file_list]
    #     eval_list = [x[0] for x in file_list]#[:2000]
    #     image_index = [name.split("_")[-1] for name in eval_list]

    # dataDir = './coco'
    # dataType = 'val'
    # annFile = '{}/annotations/instances_{}2017.json'.format(dataDir, dataType)

    # Initialize the COCO api for instance annotations
    # coco = COCO(annFile)
    # breakpoint()

    # if 'bg' in args.cam_type or 'png' in args.cam_type:
    #     iou = run_eval_cam(args, True)
    # else:
    # if args.cam_eval_thres < 1:
    #     thres_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    # else:
    #     if 'attn' in args.cam_type:
    Path(f"./COCO_IOU").mkdir(parents=True, exist_ok=True)
    Path(f"{args.save_path}").mkdir(parents=True, exist_ok=True)
    Path(f"{args.eval_mask_path}").mkdir(parents=True, exist_ok=True)
    Path(f"Check_crf_cocothing/").mkdir(parents=True, exist_ok=True)
    Path(f"CRF_refined_IOU").mkdir(parents=True, exist_ok=True)
    Path(f"Qualitative_result_{args.data_type}").mkdir(parents=True, exist_ok=True)

    # no_blocknum_count = 0

    ## Check coco classname annotation

    dataType_stuff = 'stuff_val2017'
    annFile_stuff = '/home/letitiabanana/LAVIS/coco/annotations/{}.json'.format(dataType_stuff)
    dataType_thing = 'val2017'
    annFile_thing = '/home/letitiabanana/LAVIS/coco/annotations/instances_{}.json'.format(dataType_thing)

    # print("1492", type(annFile_thing), annFile_stuff)
    # annFile =
    # initialize COCO api for instance annotations
    coco_stuff = COCO(annFile_stuff)
    # display COCO categories and supercategories
    cats_stuff = coco_stuff.loadCats(coco_stuff.getCatIds())
    nms_stuff = [cat['name'] for cat in cats_stuff]

    coco_thing = COCO(annFile_thing)
    # display COCO categories and supercategories
    cats_thing = coco_thing.loadCats(coco_thing.getCatIds())
    nms_thing = [cat['name'] for cat in cats_thing]
    # print('COCO categories: \n{}\n'.format(' '.join(nms)))
    if args.data_type == "cocoall":
        cats = cats_thing  + cats_stuff
        nms = nms_thing  + nms_stuff
    elif args.data_type == "cocothing":
        cats = cats_thing #+ cats_stuff
        nms = nms_thing #+ nms_stuff
    print("1492", type(coco_thing), type(cats_thing), type(nms_thing))
    print("1501  len(cats), len(nms)", len(cats), len(nms))

    is_coco = 'coco' in args.img_root and "stuff" not in dataType_stuff
    is_coco_stuff = 'coco' in args.img_root and "stuff" in dataType_stuff

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
            iou, mean_iou, no_blocknum_count = run_eval_cam(args, coco_stuff, coco_thing, cats, block_num,
                                                            cam_threshold=args.cam_threshold, print_log=False,
                                                            is_coco=is_coco, is_coco_stuff=is_coco_stuff)
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
