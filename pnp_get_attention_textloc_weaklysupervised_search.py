import argparse
import json

from lavis.models import load_model_and_preprocess


from lavis.models.blip_models.blip_image_text_matching import compute_gradcam, compute_gradcam_ensemble
import numpy as np
from Dataset_coco_textloc import CocoDetection
from pathlib import Path
from PIL import Image
import torch


import utils
import os

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import multiprocessing
import pickle
from statistics import mean

from skimage import transform as skimage_transform
from tqdm import tqdm

from pycocotools.coco import COCO

import time
from scipy.special import softmax

import cv2

print("Number of cpu : ", multiprocessing.cpu_count())


def ddp_setup(args, rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port
    init_process_group(backend="nccl", rank=rank,
                       world_size=world_size)  # nccl: nividia library used for distributed communication across cuda gpus


def get_args_parser():
    parser = argparse.ArgumentParser('image caption localization with ITM', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--gen_multiplecap_withpnpvqa', default="label")
    parser.add_argument('--save_path', default="Eval_test_ddp")
    parser.add_argument('--master_port', default="12355")
    parser.add_argument('--existing_att_path', default="./Cbatch_Eval_test_ddp_0521_labelascaption/img_att_forclasses/")
    parser.add_argument("--cam_out_dir", default="./Eval_test_ddp_0331/img_att_forclasses/", type=str)
    parser.add_argument("--del_patch_num", default=None,
                        help="x number of highest att patches to drop (x for each class) in one image, sort_threshxx = record all the patches above threhold xx")
    parser.add_argument("--max_att_block_num", default=10, type=int,
                        help="from which transformer layer to select the highest att patches, start from 1-12")
    # parser.add_argument("--att_loss_path",
    #                     default="./Cbatch_Eval_test_ddp_0521_labelascaption",
    #                     type=str, help="from which transformer layer to select the highest att patches")
    parser.add_argument("--img_size", default=768, type=int,
                        help="input inmage size")
    parser.add_argument("--world_size", default=4, type=int,
                        help="number of gpu used")
    parser.add_argument("--ensemble_blocks", default=None, type=str,
                        help="the attention layers used to get gradcam, start from 0 or the layer you want to first extract att with")
    parser.add_argument("--drop_iter", default=10, type=int,
                        help="the iterations of dropping patches, start from 1")
    parser.add_argument("--prune_att_head", default=None,
                        help="prune transformer attention head, indicate the retained heads, start from 0-11")
    parser.add_argument("--sort_threshold", default=None, type=float,
                        help="when dropping patches, drop all the patch that have att above this threshold")
    parser.add_argument("--edge_map_for_clip", action="store_true",
                        help="whether using edge map to calculate clip similarity")
    parser.add_argument("--final_att_threshold", default=0.05,
                        help="minimum att threshold for valid prediction in evaluation")
    parser.add_argument("--search", default=None,
                        help="searching for best hyperparameters with gt class label known for Wsupervised")
    parser.add_argument("--layer", default=None, type=str,
                        help="searching space for layer")
    parser.add_argument("--cal_token_sim_forall_layerhead", action="store_true",
                        help="argument for calculate token_contrast sum for all layer-head")
    parser.add_argument("--in_the_wild", action="store_true",
                        help="ablation test other images")
    parser.add_argument("--data_type", default=None, type=str,
                        help="cocothing or cocoall")

    return parser


def cal_clip_sim(sample, model_clip):
    with torch.inference_mode():
        clip_features = model_clip.extract_features(sample)
        image_features = clip_features.image_embeds_proj
        text_features = clip_features.text_embeds_proj
        # print("1162 text_features, image_features sizes",  (image_features @ text_features.t()).size(), image_features.size(), text_features.size()) #image_features sizes torch.Size([20, 10]) torch.Size([20, 768]) torch.Size([10, 768])

    sims = (image_features @ text_features.t()) / 0.01  # img x class
    probs_oneimg_from_clip = torch.nn.Softmax(dim=-1)(sims)  # .tolist()
    # print("1166", probs_oneimg_from_clip.shape) #img(class*2) x class
    return probs_oneimg_from_clip


def get_clipsim_for_pnmask(final_input_clip, text_for_clip, model_clip, gt_class_name):

    sample = {"image": final_input_clip, "text_input": text_for_clip}
    probs_oneimg_from_clip = cal_clip_sim(sample,
                                          model_clip).detach().cpu().numpy()  # dim  img(gtclassnum*2) x class
    # print("1223", probs_oneimg_from_clip.shape, probs_oneimg_from_clip[:len(gt_class_name),:].shape,  probs_oneimg_from_clip[len(gt_class_name):, :].shape)
    probs_posneg_img_list = np.diagonal(probs_oneimg_from_clip[:len(gt_class_name), :]).tolist() + np.diagonal(
        probs_oneimg_from_clip[len(gt_class_name):, :]).tolist()

    pos_win_sum = sum([a > b for a, b in zip(probs_posneg_img_list[:len(gt_class_name)],
                                             probs_posneg_img_list[len(gt_class_name):])]) #-1 because adding "black image" class
    # print("283 pos_win_sum", pos_win_sum)
    avg_negoverpos_clipsim = mean([a / b for a, b in zip(probs_posneg_img_list[len(gt_class_name):],
                                                         probs_posneg_img_list[:len(gt_class_name)])])  # neg/pos


    return avg_negoverpos_clipsim, pos_win_sum




def read_list(caption_path):
    # for reading also binary mode is important
    with open(caption_path, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray((x * 255).astype(np.uint8))  # Image.fromarray intake [height, width, 3]


def captions_text_loc(args, coco_thing, coco_stuff, nms, model_textloc, data_loader, vis_processors_textloc,
                      text_processors_textloc, rank, cats):
    metric_logger = utils.MetricLogger(delimiter="  ")

    print_freq = 100

    for batch_id, (norm_img_sizes, norm_img0s, img0s, paths, img_ids) in tqdm(
            enumerate(metric_logger.log_every(data_loader, print_freq)), desc='coco val captions_text_loc:'):
        if batch_id > 10:
            continue
        label_ascap_list = []  # for label as caption batch iterations
        gt_class_name_list = []  # for pnpvqa and label as caption, input individual gt class name for each image to recognize the respective label tokendsfor
        for img_id in img_ids:
            img_id = int(img_id.cpu().numpy())
            # print("121 img_id", img_id, type(img_id))
            ann_ids_stuff = coco_stuff.getAnnIds(imgIds=img_id)
            target_stuff = coco_stuff.loadAnns(ann_ids_stuff)
            ann_ids_thing = coco_thing.getAnnIds(imgIds=img_id)
            target_thing = coco_thing.loadAnns(ann_ids_thing)
            if args.data_type == "cocoall":
                target = target_thing  + target_stuff
            elif args.data_type == "cocothing":
                target = target_thing

            gt_class_name = []
            for i in range(len(target)):
                if target[i]["category_id"] == 183:
                    continue
                gt_class = getClassName(target[i]["category_id"], cats)
                # print("104 gt cls", target[i]["category_id"], gt_class)
                if gt_class not in gt_class_name:
                    gt_class_name.append(gt_class)
            gt_class_name_list.append(gt_class_name)
            label_ascap_list.append("A picture of " + " ".join(nms))  # change the max_txt_length if changing caption, in both this file and blip_image_txt_matching.py file

        run_this_batch = True
        if run_this_batch == True:

            if args.gen_multiplecap_withpnpvqa == 'label':

                for drop_iter in range(args.drop_iter):
                    run_batch = 0
                    Path(
                        f"{args.save_path}/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_save/").mkdir(
                        parents=True, exist_ok=True)
                    Path(
                        f"{args.save_path}/cam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_save/").mkdir(
                        parents=True, exist_ok=True)

                    for img_id in img_ids:
                        attention_path = f"{args.save_path}/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_numsort_thresh005/drop_iter{drop_iter}/img_att_forclasses/img_{img_id}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy"

                        highest_att_path = f"{args.save_path}/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_save/highest_att_{img_id}_{args.prune_att_head}.json"

                        sum_std_dict_path = f"Token_Contrast/max_att_block_num{args.max_att_block_num}_atthead{args.prune_att_head}/Token_contrast_sum/img_{img_id}.json"


                        if not os.path.exists(sum_std_dict_path) and args.cal_token_sim_forall_layerhead:
                            run_batch += 1
                        if os.path.exists(attention_path) and os.path.exists(highest_att_path):
                            print("attention_path exist for drop iter", drop_iter)
                            run_batch += 0
                        elif os.path.exists(attention_path) and not os.path.exists(
                                highest_att_path):
                            run_batch += 1
                        elif not os.path.exists(attention_path):
                            print("attention_path not exist for drop iter", drop_iter, ", run att")
                            run_batch += 1

                    if run_batch > 0:

                        print("  185 drop iter", drop_iter)
                        if drop_iter > 0:

                            drop_patch_img_list = []
                            drop_patch_norm_img_list = []
                            for img_idx, img0 in enumerate(img0s):
                                img_id = img_ids[img_idx]
                                gt_class_name = gt_class_name_list[img_idx]
                                norm_img0 = norm_img0s[img_idx, :, :, :]
                                drop_norm_img, drop_img = drop_image_patch_with_highest_att(args, drop_iter,
                                                                                            vis_processors_textloc,
                                                                                            norm_img0, img0, img_id,
                                                                                            gt_class_name,
                                                                                            args.del_patch_num, rank)

                                drop_patch_img_list.append(drop_img)
                                drop_patch_norm_img_list.append(drop_norm_img)
                            imgs = torch.stack(drop_patch_img_list)
                            norm_imgs = torch.stack(drop_patch_norm_img_list)

                        else:
                            norm_imgs = norm_img0s.detach().clone()
                            imgs = img0s.detach().clone()
                        imgs_in = vis_processors_textloc["eval"](imgs)  # Normalize the images
                        imgs_clip_in = None
                        vis_processors_clip = None
                        model_clip = None

                        captions = label_ascap_list
                        get_grad_cam_labelascaption(batch_id, drop_iter, args, imgs_in, imgs_clip_in, norm_img_sizes,
                                                    norm_imgs, img_ids,
                                                    gt_class_name_list, captions, model_clip, vis_processors_clip,
                                                    model_textloc, text_processors_textloc,
                                                    rank, nms, cats)


def drop_image_patch_with_highest_att(args, drop_iter, vis_processors_textloc, norm_img0, img0, img_id, gt_class_name,
                                      del_patch_num, rank):
    # For now, drop simultaneously the highest att patch for all the classes of an image
    for drop in range(drop_iter):  # range start from 0 already
        att_loss_path = f"{args.save_path}/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop}/highest_att_save/highest_att_{img_id}_{args.prune_att_head}.json"
        with open(att_loss_path, 'r') as file:
            att_loss_dict = json.load(file)

        try:
            highest_att_idx = att_loss_dict[f"halving_{img_id}"]
            if isinstance(args.del_patch_num, int):
                max_patchs = highest_att_idx[-del_patch_num:]
            elif "sort_thresh" in args.del_patch_num:
                max_patchs = highest_att_idx
        except:
            print(f"{args.max_att_block_num}_{img_id}_{class_name}", att_loss_path)
            print("217 highest att sequence of this image do not exist")
            breakpoint()

        patch_num = int(int(args.img_size) / 16)
        for max_patch in max_patchs:
            max_x = (max_patch // patch_num) * 16
            max_y = (max_patch % patch_num) * 16
            img0[:, max_x:max_x + 16, max_y:max_y + 16] = 0
            norm_img0[max_x:max_x + 16, max_y:max_y + 16, :] = 0


    return norm_img0, img0



def match_classname_in_cleancaption(classname, token_id_iter, model_textloc):
    tmp_word = []
    tmp_word_id = []
    word_list = []
    for i, token_id in enumerate(token_id_iter):
        word = model_textloc.module.tokenizer.decode([token_id])
        word = word.strip("#").strip(" ")
        word_list.append(word)
        if (word in classname and len(word) > 1) or word in ["-", "s"] or word == classname[0]:  #
            if word == classname:
                tmp_word = [word]
                tmp_word_id = [i]
                break
            elif word != "and":
                if word in tmp_word and "".join(tmp_word) != "".join(classname.split(" ")):
                    pop_id = tmp_word.index(word)
                    tmp_word.pop(pop_id)
                    tmp_word.append(word)
                    tmp_word_id.pop(pop_id)
                    tmp_word_id.append(i)
                elif word not in tmp_word:
                    tmp_word.append(word)
                    tmp_word_id.append(i)

        if "".join(tmp_word) == "".join(classname.split(" ")):
            break
        elif "".join(tmp_word[-2:]) == "".join(classname.split(" ")):
            tmp_word = tmp_word[-2:]
            tmp_word_id = tmp_word_id[-2:]
            break
        elif "".join(tmp_word[-3:]) == "".join(classname.split(" ")):
            tmp_word = tmp_word[-3:]
            tmp_word_id = tmp_word_id[-3:]
            break
        elif "baseball" in tmp_word and "ball" in tmp_word:
            b_id = tmp_word.index("ball")
            tmp_word.pop(b_id)
            tmp_word_id.pop(b_id)  # .pop gives the value of the popped item
            print("204 tmp_word_id", tmp_word_id, tmp_word, b_id)
    try:
        assert "".join(tmp_word) == "".join(classname.split(" "))
    except:
        print("191 word_list", word_list, "191 tmp_word", tmp_word, "".join(tmp_word), "191 classname",
              "".join(classname.split(" ")))
        breakpoint()
    if 0 in tmp_word_id:
        tmp_word_id.pop(tmp_word.index(0))
    return tmp_word_id


def get_grad_cam_labelascaption(batch_id, drop_iter, args, imgs, imgs_clip_in, norm_img_sizes, norm_imgs, img_ids,
                                gt_class_name_list, captions,
                                model_clip, vis_processors_clip, model_textloc, text_processors_textloc, rank, nms,
                                cats):


    """#### Compute GradCam"""

    txt_tokens = model_textloc.module.tokenizer(captions, padding="max_length", max_length=500,
                                                return_tensors="pt").to(rank)


    # if args.ensemble_blocks:
    # gradcam_ensemble,  cam_ensemble, output = compute_gradcam_ensemble(args, model_textloc.module, imgs.to(rank), captions,
    #                                                     txt_tokens,
    #                                                     block_num_range=args.ensemble_blocks)
    gradcam_ensemble,  cam_ensemble, output = compute_gradcam_ensemble(args, model_textloc.module, imgs.to(rank), captions,
                                                        txt_tokens, drop_iter)
    loss_for_bacth = softmax(output.cpu().detach().numpy(), axis=0)[:, 1]  # output[:, 1]

    img_shape = imgs[0].shape[1:]
    att_loss_record = {}

    for img_idx in range(imgs.shape[0]):
        gt_class_name = gt_class_name_list[img_idx]
        ''' record ITMloss '''
        att_loss_record[f"{img_ids[img_idx]}_ITMloss"] = float(loss_for_bacth[img_idx])
        '''norm img for one img '''
        norm_img = norm_imgs[img_idx, :, :, :].squeeze(0).numpy()

        '''save clip sim for drop > 1 and get gradcam id '''
        gradcam_id_byclass_dict = {}
        for class_idx, class_name in enumerate(gt_class_name):  # 12 block only need to match name once
            token_id_iter = txt_tokens.input_ids[img_idx][1:-1]
            assert args.gen_multiplecap_withpnpvqa == 'label'
            gradcam_id_list = match_classname_in_cleancaption(class_name, token_id_iter, model_textloc)
            gradcam_id_byclass_dict[f"{class_name}"] = gradcam_id_list

        """#### GradCam for each token"""

        if args.ensemble_blocks is not None and "saveall" in args.ensemble_blocks:

            for block in range(0, 11):  # saving layer 3 and 4 only
                for head in range(0, 12):
                    max_block_num = block + 1
                    att_head = head
                    '''Saving ground truth label att map for model evaluation '''
                    # Path(
                    #     f"{args.save_path}/cam/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_save/").mkdir(
                    #     parents=True, exist_ok=True)
                    Path(
                        f"{args.save_path}/gradcam/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_save/").mkdir(
                        parents=True, exist_ok=True)
                    '''check token map, all token got attention'''
                    att_loss_record = save_img_union_attention(args, gradcam_id_byclass_dict, gt_class_name,
                                                               att_loss_record,
                                                               img_ids,
                                                               img_idx, drop_iter, norm_img, loss_for_bacth,
                                                               img_shape, cats, nms, att_head=att_head,
                                                               max_block_num=max_block_num,
                                                               cam_att=gradcam_ensemble[block][head][img_idx],
                                                               cam_type="gradcam")
                    #
                    # att_loss_record = save_img_union_attention(args, gradcam_id_byclass_dict, gt_class_name,
                    #                                            att_loss_record,
                    #                                            img_ids,
                    #                                            img_idx, drop_iter, norm_img, loss_for_bacth,
                    #                                            img_shape, cats, nms, att_head=att_head,
                    #                                            max_block_num=max_block_num,
                    #                                            cam_att=cam_ensemble[block][head][img_idx],
                    #                                            cam_type="cam")

        else:
            if args.prune_att_head:
                    max_block_num = args.max_att_block_num
                    att_head = int(args.prune_att_head)

                    # att_loss_record = save_img_union_attention(args, gradcam_id_byclass_dict, gt_class_name,
                    #                                            att_loss_record,
                    #                                            img_ids,
                    #                                            img_idx, drop_iter, norm_img, loss_for_bacth,
                    #                                            img_shape, cats, nms, att_head=att_head,
                    #                                            max_block_num=max_block_num,
                    #                                            cam_att=cam_ensemble[max_block_num-1][att_head][img_idx],
                    #                                            cam_type="cam")

                    att_loss_record = save_img_union_attention(args, gradcam_id_byclass_dict, gt_class_name,
                                                               att_loss_record,
                                                               img_ids,
                                                               img_idx, drop_iter, norm_img, loss_for_bacth,
                                                               img_shape, cats, nms, att_head=att_head,
                                                               max_block_num=max_block_num,
                                                               cam_att=gradcam_ensemble[max_block_num-1][att_head][img_idx],
                                                               cam_type="gradcam")




def save_img_union_attention(args, gradcam_id_byclass_dict, gt_class_name, att_loss_record, img_ids, img_idx, drop_iter,
                             norm_img, loss_for_bacth, img_shape, cats, nms, att_head, max_block_num=None, cam_att=None,
                             cam_type="gradcam"):
    img_att_forclasses = {}
    background_att = {}
    img_att_forclasses_tmp = None

    classes = []
    highest_att_byimage = {}

    if drop_iter > 0:
        sum_cam_by_class = cam_att[3:,:,:].sum(dim=0)
        sort_union = sum_cam_by_class.flatten().cpu().numpy().copy()
        for drop in range(drop_iter):
            att_loss_path = f"{args.save_path}/{cam_type}/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop}/highest_att_save/highest_att_{img_ids[img_idx]}_{args.prune_att_head}.json"
            with open(att_loss_path, 'r') as file:
                att_loss_dict = json.load(file)
            highest_att_idx = att_loss_dict[f"halving_{img_ids[img_idx]}"]
            for idx in highest_att_idx:
                sort_union[idx] = 0

        save_len = int(len(sort_union) / (2 ** (drop_iter + 1)))
        if isinstance(args.del_patch_num, int):
            halving_att_sort_patch = np.argsort(sort_union.flatten())[
                                     -10:]  # each union att map, get the index of the first 10 highest patch, and save the respective loss for matching image with caption
        elif "sort_thresh" in args.del_patch_num:
            halving_att_sort_patch = np.argsort(sort_union.flatten())[-save_len:]
    else:

        sum_cam_by_class = cam_att[3:,:,:].sum(dim=0)
        sort_union = sum_cam_by_class.flatten().cpu().numpy().copy()
        save_len = int(len(sort_union) / (2 ** (drop_iter + 1)))
        if isinstance(args.del_patch_num, int):
            halving_att_sort_patch = np.argsort(sort_union.flatten())[
                                     -10:]  # each union att map, get the index of the first 10 highest patch, and save the respective loss for matching image with caption
        elif "sort_thresh" in args.del_patch_num:
            halving_att_sort_patch = np.argsort(sort_union.flatten())[-save_len:]
    highest_att_byimage[f"halving_{img_ids[img_idx]}"] = halving_att_sort_patch
    combine_json_path = f"{args.save_path}/{cam_type}/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/highest_att_save/"
    with open(os.path.join(combine_json_path + f"highest_att_{img_ids[img_idx]}_{att_head}.json"),
              "w") as outfile:
        json.dump(highest_att_byimage, outfile, cls=NumpyArrayEncoder)

    for class_idx, class_name in enumerate(gt_class_name):
        gradcam_id_list = gradcam_id_byclass_dict[f"{class_name}"]

        class_gradcam_list = []
        for id in gradcam_id_list:
            class_gradcam_list.append(cam_att[id])

        if len(class_gradcam_list) < 1:
            print(img_ids[img_idx], f"217 no {class_name} objects detected")
            breakpoint()

        ###### visualize the union of all the attention map of the class name ###
        union_Att = torch.mean(torch.stack(class_gradcam_list), dim=0).numpy()
        union_Att[union_Att < 0] = 0  # after gradcam there are negative values, so need to remove
        union_Att = (union_Att - union_Att.min()) / (
                    union_Att.max() - union_Att.min())  # 后面没有union cancel 了，所以没有normalize，这里就得normalize


        '''for saving qualitative attention map '''
        # if int(img_ids[img_idx]) in [397133, 86483, 312213, 445248, 205105, 266981, 268831, 271471, 263796, 481480,
        #               153343, 92091, 483050, 509258, 437351, 312278, 267537, 205282, 443303,
        #               438017, 455157, 540414, 519764, 15278, 106563, 314294]:
        #     Path(
        #         f"{args.save_path}/{cam_type}/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/Union_check0928/").mkdir(
        #         parents=True, exist_ok=True)
        #     gradcam_image_c = getAttMap(norm_img, union_Att, blur=True)  # range between 0-1
        #     im_gradcam_c = Image.fromarray((gradcam_image_c * 255).astype(np.uint8), 'RGB')
        #     im_gradcam_c.save(
        #         f"{args.save_path}/{cam_type}/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/Union_check0928/drop_iter{drop_iter}_Class_{class_name}_UnionGradcam_img_{img_ids[img_idx]}_max_blocknum_{max_block_num}_atthead_{att_head}.jpeg")


        union_Att = np.expand_dims(union_Att, axis=0)

        if img_att_forclasses_tmp is None:
            img_att_forclasses_tmp = union_Att
        else:
            img_att_forclasses_tmp = np.concatenate((img_att_forclasses_tmp, union_Att), axis=0)


        classes.append(getClassid(class_name.split("_")[0], cats))


    if drop_iter == 0 and "background" in gt_class_name:
        gt_class_name.pop(gt_class_name.index("background"))

    try:
        assert len(classes) == len(gt_class_name)  # -2 because add background and picture into gt_class_name
    except:
        print("undetected classes exist")
        print("240 classes, len(gt_class_name)", len(classes), len(gt_class_name), classes, gt_class_name)
        breakpoint()

    img_att_forclasses["keys"] = np.array(classes)
    img_att_forclasses["att_map"] = img_att_forclasses_tmp
    if len(gt_class_name) > 1 and "topk" in gt_class_name[0]:
        Path(
            f"./{args.save_path}/{cam_type}/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/topk_img_att_forclasses").mkdir(
            parents=True,
            exist_ok=True)
        np.save(
            f"{args.save_path}/{cam_type}/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/topk_img_att_forclasses/img_{img_ids[img_idx]}_max_blocknum_{max_block_num}_atthead_{att_head}.npy",
            img_att_forclasses)
    else:
        Path(
            f"./{args.save_path}/{cam_type}/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses").mkdir(
            parents=True,
            exist_ok=True)
        np.save(
            f"{args.save_path}/{cam_type}/max_att_block_num{max_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/img_att_forclasses/img_{img_ids[img_idx]}_max_blocknum_{max_block_num}_atthead_{att_head}.npy",
            img_att_forclasses)

    return att_loss_record


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


# class_name = "cheese"
def getClassid(classname, cats):
    for i in range(len(cats)):
        if cats[i]['name'] == classname:
            return cats[i]['id']
    return "None"


def getClassName(class_id, cats):
    for i in range(len(cats)):
        # print("234 get classname", cats[i]['id']==class_id, type(cats[i]['id']), type(class_id))
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return "None"


def Attmap_resize(img_shape, attMap):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap = attMap / attMap.max()
    attMap = skimage_transform.resize(attMap, (img_shape), order=3, mode="constant")
    return attMap


def main(rank, world_size, args):
    tic = time.perf_counter()
    ddp_setup(args, rank, args.world_size)

    torch.cuda.set_device(rank)

    dataType_stuff = 'stuff_val2017'
    annFile_stuff = '/home/letitiabanana/LAVIS/coco/annotations/{}.json'.format(dataType_stuff)
    dataType_thing = 'val2017'
    annFile_thing = '/home/letitiabanana/LAVIS/coco/annotations/instances_{}.json'.format(dataType_thing)


    coco_stuff = COCO(annFile_stuff)
    # display COCO categories and supercategories
    cats_stuff = coco_stuff.loadCats(coco_stuff.getCatIds())
    nms_stuff = [cat['name'] for cat in cats_stuff]

    coco_thing = COCO(annFile_thing)
    # display COCO categories and supercategories
    cats_thing = coco_thing.loadCats(coco_thing.getCatIds())
    nms_thing = [cat['name'] for cat in cats_thing]

    if args.data_type == "cocoall":
        cats = cats_thing + cats_stuff
        nms = nms_thing + nms_stuff
    elif args.data_type == "cocothing":
        cats = cats_thing
        nms = nms_thing
    print("1492", type(coco_thing), type(cats_thing), type(nms_thing))
    print("1501  len(cats), len(nms)", len(cats), len(nms))


    """### Load coco images into batches"""

    """#### Load text localization model and preprocessors"""

    model_textloc, vis_processors_textloc, text_processors_textloc = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=rank, is_eval=True)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(count_parameters(model_textloc))
    # breakpoint()
    imageDir = '/home/letitiabanana/LAVIS/coco/images/val2017'

    test_data = CocoDetection(imageDir, args, annFile_thing, vis_processors_textloc, getClassName, cats,
                              device=rank)  ##add arguments
    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        sampler=DistributedSampler(test_data)
    )

    model_textloc = DDP(model_textloc, device_ids=[rank])

    captions_text_loc(args, coco_thing, coco_stuff, nms, model_textloc, data_loader_test,
                      vis_processors_textloc, text_processors_textloc, rank, cats)
    toc = time.perf_counter()
    print(f"Time: total running time for matching max blocknum & 100 images & 10 dropiter {toc - tic:0.4f} seconds")

    destroy_process_group()


def Search(para):
    from argparse import Namespace
    save_path = "Cbatch_Eval_test_ddp_1028_768_flickrfinetune_zeroshot_cocostuff_ablation_alllayerhead"
    args = Namespace(max_att_block_num=int(para["layer"]), prune_att_head=str(para["head"]),
                     final_att_threshold=para["min_att"], batch_size=1, img_size=336, drop_iter=1, save_path=save_path,
                     del_patch_num="sort_thresh005", search="Wsupervised", cam_type="gradcam", in_the_wild=False)
    Path(
        f"./Edge_plus_pos/").mkdir(
        parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #"cuda:3"
    # print(1390, args)
    if os.path.exists(f"Search_Wsupervised_cocostuff_flickr/layer{args.max_att_block_num}_head{args.prune_att_head}/reward_metric_dict_attthresh{int(args.final_att_threshold*100)}.json"):
        with open(
                f"Search_Wsupervised_cocostuff_flickr/layer{args.max_att_block_num}_head{args.prune_att_head}/reward_metric_dict_attthresh{int(args.final_att_threshold*100)}.json", "r") as outfile:
            reward_metric_dict = json.load(outfile)
        total_pos_mask_topklabel_score = reward_metric_dict[f"total_pos_win_sum_layer{args.max_att_block_num}_head{args.prune_att_head}"]
        print("reward exists")
        return total_pos_mask_topklabel_score

    tic1 = time.perf_counter()
    test_data = CocoDetection(imageDir, args, annFile_thing, getClassName, cats, vis_processors_clip)  ##add arguments
    torch.manual_seed(1)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        # num_workers=0,
        drop_last=False,
        shuffle=False,
        # sampler=Sampler(test_data)
    )
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 100
    total_pos_win_sum = 0
    total_pos_win_sum_scaleby_tokencontrast = 0
    reward_metric_dict = {}
    model_clip.to(device)
    for batch_id, (batch_image_input_clip, batch_text_input_clip, img_ids) in tqdm(
            enumerate(metric_logger.log_every(data_loader_test, print_freq)), desc='EVAL reward:'):

        if batch_text_input_clip[0][0] == "No gt class for the image":
            print("1431 No gt class for the image")
            continue


        sum_std_dict_path = f"Token_Contrast/max_att_block_num{args.max_att_block_num}_atthead{args.prune_att_head}_withatt{int(args.final_att_threshold * 100)}/Token_contrast_sum/img_{img_ids[0]}.json"
        with open(sum_std_dict_path, "r") as outfile:
            sum_std_dict = json.load(outfile)
        token_contrast = sum_std_dict[f"TCSum_layer{args.max_att_block_num}_head{args.prune_att_head}_img_{img_ids[0]}"]



        text_input_clip = [label[0] for label in batch_text_input_clip]
        # text_input_clip.append("black image")
        if batch_id == 0:
            print("1427", batch_image_input_clip[0].shape, text_input_clip, img_ids)
        avg_negoverpos_clipsim, pos_win_sum = get_clipsim_for_pnmask(batch_image_input_clip[0].type(torch.FloatTensor).to(device), text_input_clip, model_clip, gt_class_name=text_input_clip)
        total_pos_win_sum += pos_win_sum
        total_pos_win_sum_scaleby_tokencontrast += pos_win_sum * (1 - token_contrast)
        reward_metric_dict[f"{img_ids[0]}_pos_win_sum"] = pos_win_sum
        reward_metric_dict[f"{img_ids[0]}_pos_win_sum_scaleby_tokencontrast"] = pos_win_sum * (1 - token_contrast)


    print("1433 total_pos_win_sum", total_pos_win_sum)
    reward_metric_dict[
        f"total_pos_win_sum_layer{args.max_att_block_num}_head{args.prune_att_head}"] = total_pos_win_sum
    reward_metric_dict[
        f"total_pos_win_sum_scaleby_tokencontrast_layer{args.max_att_block_num}_head{args.prune_att_head}"] = total_pos_win_sum_scaleby_tokencontrast
    # print("1435 reward_metric_dict.items()", reward_metric_dict.items())
    sorted_reward_metric_dict = {k: v for k, v in sorted(reward_metric_dict.items(), key=lambda item: item[1])}
    Path(f"Search_Wsupervised_cocostuff_flickr/layer{args.max_att_block_num}_head{args.prune_att_head}/").mkdir(parents=True, exist_ok=True)
    with open(
            f"Search_Wsupervised_cocostuff_flickr/layer{args.max_att_block_num}_head{args.prune_att_head}/reward_metric_dict_attthresh{int(args.final_att_threshold * 100)}.json",
            "w") as outfile:
        json.dump(sorted_reward_metric_dict, outfile, cls=NumpyArrayEncoder)

    toc1 = time.perf_counter()
    print(f"Time: total running time for one search on layer{args.max_att_block_num}_head{args.prune_att_head} attthresh{int(args.final_att_threshold * 100)}: {toc1 - tic1:0.4f} seconds")

    return total_pos_win_sum







if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    torch.cuda.empty_cache()
    Path(f"Pos_mask/").mkdir(
        parents=True, exist_ok=True)
    Path(f"Search_Wsupervised_cocostuff_flickr/").mkdir(
        parents=True, exist_ok=True)
    world_size = args.world_size  # torch.cuda.device_count()


    if args.search:

        dataType_stuff = 'stuff_val2017'
        annFile_stuff = '/home/letitiabanana/LAVIS/coco/annotations/{}.json'.format(dataType_stuff)
        dataType_thing = 'val2017'
        annFile_thing = '/home/letitiabanana/LAVIS/coco/annotations/instances_{}.json'.format(dataType_thing)


        coco_stuff = COCO(annFile_stuff)
        # display COCO categories and supercategories
        cats_stuff = coco_stuff.loadCats(coco_stuff.getCatIds())
        nms_stuff = [cat['name'] for cat in cats_stuff]

        coco_thing = COCO(annFile_thing)
        # display COCO categories and supercategories
        cats_thing = coco_thing.loadCats(coco_thing.getCatIds())
        nms_thing = [cat['name'] for cat in cats_thing]
        # print('COCO categories: \n{}\n'.format(' '.join(nms)))
        cats = cats_thing  + cats_stuff
        nms = nms_thing  + nms_stuff
        print("1492", type(coco_thing), type(cats_thing), type(nms_thing))
        print("1501  len(cats), len(nms)", len(cats), len(nms))
        # breakpoint()

        """### Load coco images into batches"""

        """#### Load text localization model and preprocessors"""

        model_clip, vis_processors_clip, txt_processors_clip = load_model_and_preprocess("clip_feature_extractor",
                                                                                         model_type="ViT-L-14-336",
                                                                                         is_eval=True)

        imageDir = '/home/letitiabanana/LAVIS/coco/images/val2017'

        print("modify args namespace again!!!!")
        from Gradient_Free_Optimizers_master.gradient_free_optimizers import BayesianOptimizer, RandomSearchOptimizer, GridSearchOptimizer,RandomRestartHillClimbingOptimizer,RandomAnnealingOptimizer
        layer_start = int(args.layer.split("-")[0])
        layer_end = int(args.layer.split("-")[1])
        search_space = {
            "layer": np.arange(layer_start, layer_end+1),

            "head": np.arange(0, 12),
            "min_att": np.arange(0.05, 0.5, 0.1)
            # "layer": np.arange(8, 10),
            # "head": np.arange(9, 10),
            # "min_att": np.arange(0.05, 0.5, 0.1)
        }



        opt = RandomAnnealingOptimizer(search_space, random_state=1164005944)
        opt.search(Search, n_iter=34)

    else:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
