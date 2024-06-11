import argparse
import json

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess

from Load_datasets import load_voc, load_psc, load_ade20k
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam, compute_gradcam_ensemble
import numpy as np

from pathlib import Path
from PIL import Image
import torch

import utils
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from os import walk

import multiprocessing

from statistics import mean


from scipy.ndimage import filters
from skimage import transform as skimage_transform
from tqdm import tqdm

from pycocotools.coco import COCO

import time


# for edge map
import cv2
import torch.nn.functional as F


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
    parser.add_argument('--home_dir', default="/home/letitiabanana/LAVIS/")
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
                        help="voc, psc, ade20k, cocothing or cocoall")
    parser.add_argument("--postprocess", default=None, type=str,
                        help="blur or crf or blur+crf")
    parser.add_argument("--threshold", default=None,  type=float,
                        help="attention map to binary map threshold")

    return parser



def token_cos_sim_map(img_ids, inputs, layer, head, with_att_threshold=None):  # 算的是本身的token contrast，暂时与sort_threshold无关
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
        if img_ids[i] in [86483, 312213, 445248, 205105, 266981, 268831, 271471, 263796, 481480,
                  153343, 92091, 483050, 509258, 437351, 312278, 267537, 205282, 443303,
                  438017, 455157, 540414, 519764, 15278, 106563, 314294]:
            cos_sim_map = Image.fromarray((by_img_att * 255).astype(np.uint8), 'L')
            cos_sim_map.save(
                f"Token_Contrast/max_att_block_num{layer}_atthead{head}_withatt{with_att_threshold}/Cos_sim_map/img_{img_ids[i]}.jpeg")
        std = np.std(by_img_att)
        sum_token_sim = np.sum(by_img_att) /(by_img_att.shape[0]*by_img_att.shape[1])
        sum_std_dict[f"TCSum_layer{layer}_head{head}_img_{img_ids[i]}"] = sum_token_sim
        sum_std_dict[f"TCStd_layer{layer}_head{head}_img_{img_ids[i]}"] = std
        sum_std_dict_path = f"Token_Contrast/max_att_block_num{layer}_atthead{head}_withatt{with_att_threshold}/Token_contrast_sum/img_{img_ids[i]}.json"
        # if os.path.exists(sum_std_dict_path):
        #     with open(sum_std_dict_path,"r") as outfile:
        #         sum_before = json.load(outfile)
        #     sum_before.update(sum_std_dict)
        #     sum_std_dict_save = sum_before
        # else:
        #     sum_std_dict_save = sum_std_dict
        with open(sum_std_dict_path,"w") as outfile:
            json.dump(sum_std_dict, outfile, cls=NumpyArrayEncoder)



        # print("525 std", std)





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





"""### Load question and coco categories"""



def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray((x * 255).astype(np.uint8))  # Image.fromarray intake [height, width, 3]


def captions_text_loc(args,  nms, model_textloc, data_loader, vis_processors_textloc,
                      text_processors_textloc, rank, cats):
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 100

    for batch_id, (org_img_sizes, norm_img0s, img0s, imgs_for_clip, imgs_clip_large_in, paths, img_ids, label) in tqdm(
            enumerate(metric_logger.log_every(data_loader, print_freq)), desc='VOC val captions_text_loc:'):
        #

        print("116 batch stats", norm_img0s.shape, type(paths), img0s.shape,
              img_ids)  # torch.Size([batch_size, 384, 384, 3]), <class 'tuple'>,  torch.Size([batch_size, 3, 384, 384]),  10

        label_ascap_list = []  # for label as caption batch iterations
        gt_class_name_list = []  # for pnpvqa and label as caption, input individual gt class name for each image to recognize the respective label tokendsfor
        for idx, img_id in enumerate(img_ids):
            img_id = int(img_id)

            gt_class_name = []
            for i in label[idx].tolist():
                if i != 0:
                    gt_class = cats[i]
                    # print("104 gt cls", target[i]["category_id"], gt_class)
                    if gt_class not in gt_class_name:
                        gt_class_name.append(gt_class)
            gt_class_name_list.append(gt_class_name)
            label_ascap_list.append("A picture of " + " ".join(
                nms))  # change the max_txt_length if changing caption, in both this file and blip_image_txt_matching.py file


        drop_iter = args.drop_iter
        tic_drop = time.perf_counter()
        norm_imgs = norm_img0s.detach().clone()
        imgs = img0s.detach().clone()
        imgs_in = imgs
        toc_drop = time.perf_counter()
        print(f"Count 1 {img_ids} Time: drop image patches use {toc_drop - tic_drop:0.4f} seconds")


        captions = label_ascap_list
        get_grad_cam_labelascaption(batch_id, drop_iter, args, imgs_in, org_img_sizes,
                                    norm_imgs, img_ids,
                                    gt_class_name_list, captions,
                                    model_textloc,
                                    rank, nms, cats)



def get_grad_cam_labelascaption(batch_id, drop_iter, args, imgs_in, org_img_sizes,
                                            norm_imgs, img_ids,
                                            gt_class_name_list, captions,
                                            model_textloc,
                                            rank, nms, cats):
    Path(
        f"./{args.save_path}/gradcam/max_att_block_num{args.max_att_block_num}_del_patch_num{args.del_patch_num}/drop_iter{drop_iter}/edge_map").mkdir(
        parents=True, exist_ok=True)



    """#### Compute GradCam drop 1-5"""

    txt_tokens = model_textloc.module.tokenizer(captions, padding="max_length", max_length=500,
                                                return_tensors="pt").to(rank)


    img_shape = imgs_in[0].shape[1:] #[768,768]

    if args.prune_att_head:

        max_block_num = args.max_att_block_num
        att_head = int(args.prune_att_head)

        save_img_union_attention(model_textloc, imgs_in, org_img_sizes, args, gt_class_name_list, img_ids, drop_iter, norm_imgs,
                                                   img_shape, cats, nms, txt_tokens, rank, att_head=att_head,
                                                   max_block_num=max_block_num,
                                                   cam_type="gradcam")




def save_img_union_attention(model_textloc, imgs_in, org_img_sizes, args, gt_class_name_list, img_ids, drop_iter,
                             norm_imgs, img_shape, cats, nms, txt_tokens, rank, att_head,  max_block_num=None,
                             cam_type="gradcam"):

    for block_num in range(7, 9):
        # 8th layer 1-12 layers so select the 7th starting from 0
        model_textloc.module.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.save_attention = False
    with torch.inference_mode():

        org_img_list = load_OrgImage(args, img_ids)
        label_trues = Load_GroundTruth(args, img_ids)

        # Filter the maps that have high mIoU with the map of class detected by CLIP
        caption_filtered_list = []
        class_filtered_list = []
        label_preds_withfiltering = []
        label_preds_withfiltering_beforeargmax = []
        best_class_idx_list = []

        for img in range(args.batch_size):
                        # get the predicted label from clip(larger then top 5)
            best_class_idx_list, class_filtered_list, caption_filtered_list = Load_predicted_classes(args, nms, best_class_idx_list, class_filtered_list, caption_filtered_list, gt_class_name_list, img_ids, img, pred_path='BLIP_0514_VOC_Clip_classification')


    '''Inference BLIP again after with the final pred'''
    txt_tokens_filtered = model_textloc.module.tokenizer(caption_filtered_list, padding="max_length",
                                                         max_length=500,
                                                         return_tensors="pt").to(rank)
    gradcam_0_filtered, gradcam_agg_dropfiltered = Inference_BLIP_filteredcaption(args, model_textloc, txt_tokens_filtered, imgs_in, norm_imgs, img_ids, caption_filtered_list, class_filtered_list, rank)

    with torch.inference_mode():
        '''for BLIP 1 drop reinference '''
        # '''Merge tokens and Draw attention maps of filtered calsses -- per image base as we now have differnet captions'''
        filtered_average_attention_map_list = []
        for img_num, gradcam_filtered in enumerate(gradcam_0_filtered): #[0][0] stands for getting 8th layer 9th head
            perimg_filtered_average_attention_map = Mean_over_filtered_label_tokens(model_textloc, txt_tokens_filtered, gradcam_filtered, class_filtered_list, img_num)
            filtered_average_attention_map_list.append(perimg_filtered_average_attention_map)



        '''Check Blip 2nd round pred with filtered caption'''
        # Threshold the map and get argmax
        label_preds_withfiltered_caption = []
        label_preds_withfiltered_caption_beforeargmax = []
        for img, pred_map in enumerate(filtered_average_attention_map_list):
            thresholded_pred_map = pred_map.clone().detach()
            for i in range(pred_map.shape[0]):
                # print(682, i, att[i].min(), att[i].max())
                thresholded_pred_map[i] = (pred_map[i] - pred_map[i].min()) / (
                            pred_map[i].max() - pred_map[i].min())
            thresholded_pred_map = (thresholded_pred_map >= args.threshold).type(torch.bool)
            Blip_final_pred = pred_map * thresholded_pred_map
            # resize the pred map

            Blip_final_pred = torch.nn.functional.interpolate(Blip_final_pred.unsqueeze(0),
                                                         size=(label_trues[img].shape[0], label_trues[img].shape[1]),
                                                         mode='bilinear', align_corners=True).squeeze()
            # !!! should add a threshold again here???
            Blip_final_pred = Scale_0_1(Blip_final_pred)
            label_preds_withfiltered_caption_beforeargmax.append(Blip_final_pred)
            if len(Blip_final_pred.shape) < 3:
                Blip_max_map = Blip_final_pred
                Blip_final_pred = Blip_final_pred.unsqueeze(0)
            else:
                Blip_max_map = torch.max(Blip_final_pred, dim=0)[0]
            background = ((Blip_max_map == 0) ).unsqueeze(0)
            # background = (1 - Blip_max_map).unsqueeze(0)

            '''For object dataset, add the background map, for context dataset, add the background map only when there are less then 3 objects in the prediction'''
            if args.data_type == "voc":
                Blip_final_pred_wbackground = torch.cat((background, Blip_final_pred), dim=0)
            elif args.data_type == "psc" or args.data_type == "ade20k":
                if len(best_class_idx_list[img]) < 3:
                    Blip_final_pred_wbackground = torch.cat((background, Blip_final_pred), dim=0)
                elif len(best_class_idx_list[img]) >= 3:
                    Blip_final_pred_wbackground = Blip_final_pred


            if args.postprocess:
                Blip_final_pred_argmax_map = postprocess(args, Blip_final_pred_wbackground, org_img_list, label_trues,
                                                         img)
            else:
                #argmax map after blip pred with filtered caption
                Blip_final_pred_argmax_map = (torch.argmax(Blip_final_pred_wbackground, dim=0)).numpy()


            if args.data_type == "voc":
                for i in range(len(best_class_idx_list[img]) - 1, -1, -1):
                    Blip_final_pred_argmax_map[Blip_final_pred_argmax_map == int(i + 1)] = best_class_idx_list[img][i]+1
            elif args.data_type == "psc" or args.data_type == "ade20k":
                if len(best_class_idx_list[img]) < 3:
                    for i in range(len(best_class_idx_list[img]) - 1, -1, -1):
                        Blip_final_pred_argmax_map[Blip_final_pred_argmax_map == int(i+1)] = best_class_idx_list[img][i]+1
                elif len(best_class_idx_list[img]) >= 3:
                    for i in range(len(best_class_idx_list[img]) - 1, -1, -1):
                        Blip_final_pred_argmax_map[Blip_final_pred_argmax_map == int(i)] = best_class_idx_list[img][i] + 1



            label_preds_withfiltered_caption.append(Blip_final_pred_argmax_map)
            Draw_Segmentation_map(args, Blip_final_pred_argmax_map, label_trues, org_img_list, img_ids, img,
                                  filename='BLIP_1_drop')


        if gradcam_agg_dropfiltered is not None:
            '''for BLIP N drop reinference '''
            # '''Merge tokens and Draw attention maps of filtered calsses -- per image base as we now have differnet captions'''
            filtered_Ndrop_attention_map_list = []
            for img_num, gradcam_filtered_agg_drop in enumerate(gradcam_agg_dropfiltered): #[0][0] stands for getting 8th layer 9th head
                perimg_filtered_Ndrop_attention_map = Mean_over_filtered_label_tokens(model_textloc,
                                                                                        txt_tokens_filtered,
                                                                                        gradcam_filtered_agg_drop,
                                                                                        class_filtered_list, img_num)
                filtered_Ndrop_attention_map_list.append(perimg_filtered_Ndrop_attention_map)


            # '''Check Blip 2nd round pred with filtered caption'''
            # Threshold the map and get argmax
            label_preds_withfiltered_caption_alldrop = []
            label_preds_withfiltered_caption_beforeargmax_alldrop = []
            for img, drop_pred_map in enumerate(filtered_Ndrop_attention_map_list):
                thresholded_drop_map = drop_pred_map.clone().detach()
                for i in range(drop_pred_map.shape[0]):
                    # print(682, i, att[i].min(), att[i].max())
                    thresholded_drop_map[i] = (drop_pred_map[i] - drop_pred_map[i].min()) / (
                                drop_pred_map[i].max() - drop_pred_map[i].min())
                thresholded_drop_map = (thresholded_drop_map >= args.threshold).type(torch.bool)

                # Blip_final_pred_drop = drop_pred_map * thresholded_drop_map
                Blip_final_pred_drop = drop_pred_map * thresholded_drop_map#thresholded_drop_map.type(torch.float32)
                # resize the pred map
                Blip_final_pred_drop = torch.nn.functional.interpolate(Blip_final_pred_drop.unsqueeze(0),
                                                             size=(label_trues[img].shape[0], label_trues[img].shape[1]),
                                                             mode='bilinear', align_corners=True).squeeze()
                # Blip_final_pred_drop = Scale_0_1(Blip_final_pred_drop)

                if len(Blip_final_pred_drop.shape) < 3:
                    Blip_max_map_drop = Blip_final_pred_drop
                    Blip_final_pred_drop = Blip_final_pred_drop.unsqueeze(0)
                else:
                    Blip_max_map_drop = torch.max(Blip_final_pred_drop, dim=0)[0]
                label_preds_withfiltered_caption_beforeargmax_alldrop.append(Blip_final_pred_drop)
                background = ((Blip_max_map_drop == 0) ).unsqueeze(0)
                # background = (1 - Blip_final_pred_drop).unsqueeze(0)
                '''For object dataset, add the background map, for context dataset, add the background map only when there are less then 3 objects in the prediction'''
                if args.data_type == "voc":
                    Blip_final_pred_wbackground_drop = torch.cat((background, Blip_final_pred_drop), dim=0)
                elif args.data_type == "psc" or args.data_type == "ade20k":
                    if len(best_class_idx_list[img]) < 3:
                        Blip_final_pred_wbackground_drop = torch.cat((background, Blip_final_pred_drop), dim=0)
                    elif len(best_class_idx_list[img]) >= 3:
                        Blip_final_pred_wbackground_drop = Blip_final_pred_drop



                # #### Blur + CRF
                if args.postprocess:
                    Blip_final_pred_argmax_map_drop = postprocess(args, Blip_final_pred_wbackground_drop, org_img_list,
                                                                  label_trues, img)
                else:
                    #argmax map after blip pred with filtered caption
                    Blip_final_pred_argmax_map_drop = (torch.argmax(Blip_final_pred_wbackground_drop, dim=0)).numpy()


                if args.data_type == "voc":
                    for i in range(len(best_class_idx_list[img]) - 1, -1, -1):
                        Blip_final_pred_argmax_map_drop[Blip_final_pred_argmax_map_drop == int(i + 1)] = \
                        best_class_idx_list[img][i] + 1
                elif args.data_type == "psc" or args.data_type == "ade20k":
                    if len(best_class_idx_list[img]) < 3:
                        for i in range(len(best_class_idx_list[img]) - 1, -1, -1):
                            Blip_final_pred_argmax_map_drop[Blip_final_pred_argmax_map_drop == int(i + 1)] = \
                            best_class_idx_list[img][i] + 1
                    elif len(best_class_idx_list[img]) >= 3:
                        for i in range(len(best_class_idx_list[img]) - 1, -1, -1):
                            Blip_final_pred_argmax_map_drop[Blip_final_pred_argmax_map_drop == int(i)] = best_class_idx_list[img][
                                                                                                   i] + 1


                # for i, class_id in enumerate(class_filtered_id_list[img]):
                #     Blip_final_pred_argmax_map_drop[Blip_final_pred_argmax_map_drop==int(i+1)] = cats[class_id]['id']
                label_preds_withfiltered_caption_alldrop.append(Blip_final_pred_argmax_map_drop)
                Draw_Segmentation_map(args, Blip_final_pred_argmax_map_drop, label_trues, org_img_list, img_ids, img,
                                      filename = 'BLIP_N_drop')




        '''Calculate mIoU'''

        if gradcam_0_filtered is not None:
            acc_table_withfiltered_caption, hist_withfiltered_caption = scores(label_trues, label_preds_withfiltered_caption, cats,n_class=len(cats)+1)
            print(img_ids, "miou filtered_caption", acc_table_withfiltered_caption['Mean IoU'])

        if gradcam_agg_dropfiltered is not None:
            acc_table_all_drop_with_filtered_caption, hist_all_drop_with_filtered_caption = scores(label_trues, label_preds_withfiltered_caption_alldrop, cats,
                                                       n_class=len(cats)+1)
            print(img_ids, "miou all_drop_with_filtered caption", acc_table_all_drop_with_filtered_caption['Mean IoU'])


        Path(
            f"{args.save_path}/hist_withfiltered_caption/").mkdir(
            parents=True, exist_ok=True)
        Path(
            f"{args.save_path}/all_drop_hist_with_filtered_caption/").mkdir(
            parents=True, exist_ok=True)


        if gradcam_0_filtered is not None:
            np.save(
                f"{args.save_path}/hist_withfiltered_caption/img_{img_ids[0]}_max_blocknum_{max_block_num}_atthead_{att_head}.npy",
                hist_withfiltered_caption)
        if gradcam_agg_dropfiltered is not None:
            np.save(
                f"{args.save_path}/all_drop_hist_with_filtered_caption/img_{img_ids[0]}_max_blocknum_{max_block_num}_atthead_{att_head}.npy",
                hist_all_drop_with_filtered_caption)
        return None



    ######################## filter label based onbest_class_idx = probs_oneimg_from_clip_all.topk(1)[1].unique()clip similarity ######################

def Full_Filter_Ensemble(args, cats, label_trues, org_img_list, best_class_idx_list, Preds_withfiltered_caption_beforeargmax, Full_preds_withfiltering_beforeargmax):
    Preds_ensemble_list = []
    for img, (i, j) in enumerate(
            zip(Preds_withfiltered_caption_beforeargmax, Full_preds_withfiltering_beforeargmax)):
        ensemble_map = (i + j) / 2
        if len(ensemble_map.shape) < 3:
            ensemble_max_map = ensemble_map
            ensemble_map = ensemble_map.unsqueeze(0)
        else:
            ensemble_max_map = torch.max(ensemble_map, dim=0)[0]
        background = ((ensemble_max_map == 0)).unsqueeze(0)
        # background = (1 - ensemble_max_map).unsqueeze(0)
        ensemble_final_pred_wbackground = torch.cat((background, ensemble_map), dim=0)
        if args.postprocess:
            if "blur" in args.postprocess:
                ensemble_final_pred_wbackground_blur_list = []
                for i in range(ensemble_final_pred_wbackground.shape[0]):
                    ensemble_final_pred_wbackground_blur = torch.from_numpy(
                        blurring(ensemble_final_pred_wbackground[i],
                                 (label_trues[img].shape[0], label_trues[img].shape[1]), scale=0.05))
                    ensemble_final_pred_wbackground_blur_list.append(ensemble_final_pred_wbackground_blur)
                ensemble_final_pred_wbackground = torch.stack(ensemble_final_pred_wbackground_blur_list, axis=0)
            if "crf" in args.postprocess:
                ensemble_final_pred_argmax_map = densecrf(org_img_list[img], ensemble_final_pred_wbackground)
        else:
            # argmax map ensemble
            ensemble_final_pred_argmax_map = (torch.argmax(ensemble_final_pred_wbackground, dim=0)).numpy()

        for i in range(len(best_class_idx_list[img]) - 1, -1, -1):
            ensemble_final_pred_argmax_map[ensemble_final_pred_argmax_map == int(i + 1)] = best_class_idx_list[img][i]+1
        # for i, class_id in enumerate(class_filtered_id_list[img]):
        #     ensemble_final_pred_argmax_map[ensemble_final_pred_argmax_map==int(i+1)] = cats[class_id]['id']

        Preds_ensemble_list.append(ensemble_final_pred_argmax_map)
    return Preds_ensemble_list


def Inference_BLIP_filteredcaption(args, model_textloc, txt_tokens_filtered, imgs_in, norm_imgs, img_ids, caption_filtered_list, class_filtered_list, rank):
    if args.drop_iter == 1:

        gradcam_filterd_ensemble, cam_filterd_ensemble, filterd_output = compute_gradcam_ensemble(args,
                                                                                                  model_textloc.module,
                                                                                                  imgs_in.to(rank),
                                                                                                  caption_filtered_list,
                                                                                                  txt_tokens_filtered)
        layer = int(args.max_att_block_num) - 1
        head = int(args.prune_att_head)
        gradcam_0_filtered = gradcam_filterd_ensemble[layer][head].detach().clone()
        gradcam_agg_dropfiltered = None

    else:
        imgs_in_for_drop = imgs_in.detach().clone()
        norm_imgs_for_drop = norm_imgs.detach().clone()
        max_patches_per_image = {}
        for img, img_id in enumerate(img_ids):
            max_patches_per_image[int(img_id)] = []
            # print(1150, img_id, imgs_in_for_drop[img].sum())

        norm_imgs_for_drop_list = []
        gradcam_ensemble_list = []
        for drop_iter in range(args.drop_iter):
            # Drop the org img with the idx from salience drop
            for img, img_id in enumerate(img_ids):
                # print(1157,"before drop", drop_iter, img_id, imgs_in_for_drop[img].sum())
                # print(1157, "before drop norm", drop_iter, img_id, norm_imgs_for_drop[img].sum())
                # img_in_for_drop = imgs_in_for_drop[img]
                # norm_img_for_drop = norm_imgs_for_drop[img]
                max_patches = max_patches_per_image[int(img_id)]
                patch_num = int(int(args.img_size) / 16)
                dropped_value = 0
                for max_patch in max_patches:
                    max_x = (max_patch // patch_num) * 16
                    max_y = (max_patch % patch_num) * 16
                    dropped_value += imgs_in_for_drop[img, :, max_x:max_x + 16, max_y:max_y + 16].sum()
                    # print("img[max_x:max_x+16, max_y:max_y+16] before", max_x, max_x + 16, max_y, max_y + 16, img0[:, max_x:max_x + 16, max_y:max_y + 16].sum(), norm_img0[max_x:max_x + 16, max_y:max_y + 16, :].sum(), img0.sum())
                    imgs_in_for_drop[img, :, max_x:max_x + 16, max_y:max_y + 16] = 0
                    norm_imgs_for_drop[img, max_x:max_x + 16, max_y:max_y + 16, :] = 0
                # print(1171, "after drop", drop_iter, dropped_value, img_id, imgs_in_for_drop[img].sum())
                # print(1171, "after drop norm", drop_iter, img_id, norm_imgs_for_drop[img].sum())
            norm_imgs_for_drop_list.append(norm_imgs_for_drop.detach().clone())
            # Get the gradcam map of the dropped image and complete caption
            txt_tokens_filtered = model_textloc.module.tokenizer(caption_filtered_list, padding="max_length",
                                                                 max_length=500,
                                                                 return_tensors="pt").to(rank)
            gradcam_filterd_ensemble, cam_filterd_ensemble, filterd_output = compute_gradcam_ensemble(args,
                                                                                                      model_textloc.module,
                                                                                                      imgs_in_for_drop.to(
                                                                                                          rank),
                                                                                                      caption_filtered_list,
                                                                                                      txt_tokens_filtered,
                                                                                                      drop_iter)

            layer = int(args.max_att_block_num) - 1
            head = int(args.prune_att_head)
            gradcam_ensemble_lh = gradcam_filterd_ensemble[layer][head].detach().clone()

            gradcam_ensemble_lh_pred = gradcam_ensemble_lh.detach().clone()
            for img, img_id in enumerate(img_ids):
                dropped_gradcam_value = 0
                max_patches = max_patches_per_image[int(img_id)]
                # print(1188, "filtered gradcam value before drop", img_id,  gradcam_ensemble_lh_pred[img].sum())
                for max_patch in max_patches:  # Here we are dropping the attention map not the org image, so do not need to multiply by 16
                    max_x = (max_patch // patch_num)
                    max_y = (max_patch % patch_num)
                    dropped_gradcam_value += gradcam_ensemble_lh_pred[img][:, max_x, max_y].sum()
                    # print("img[max_x:max_x+16, max_y:max_y+16] before", max_x, max_x + 16, max_y, max_y + 16, img0[:, max_x:max_x + 16, max_y:max_y + 16].sum(), norm_img0[max_x:max_x + 16, max_y:max_y + 16, :].sum(), img0.sum())
                    gradcam_ensemble_lh_pred[img][:, max_x, max_y] = 0
                # print(1195, "filtered gradcam value after drop", img_id, dropped_gradcam_value, gradcam_ensemble_lh_pred[img].sum())
            gradcam_ensemble_list.append(gradcam_ensemble_lh_pred)

            # Given the gradcam, get the id of the top 50% of patches to drop
            for img, img_id in enumerate(img_ids):
                sum_cam_by_class = gradcam_ensemble_lh[img][3:-1, :, :].sum(dim=0)
                sort_union = sum_cam_by_class.flatten().cpu().numpy().copy()
                for idx in max_patches_per_image[int(img_id)]:
                    sort_union[idx] = 0
                save_len = 10 #int(len(sort_union) / ( 2** (drop_iter + 1)))
                # print(537, save_len)
                if "sort_thresh" in args.del_patch_num:
                    halving_att_sort_patches = np.argsort(sort_union.flatten())[-save_len:]
                max_patches_per_image[int(img_id)].extend(halving_att_sort_patches)
            # drop the attention on patches that are already dropped

        '''Draw the attention map for 1st iteration and for later drops'''
        for drop_iter in range(args.drop_iter):
            '''Merge tokens and Draw attention maps of filtered calsses -- per image base as we now have differnet captions'''
            filtered_average_attention_map_list = []
            for img_num, gradcam_filtered in enumerate(
                    gradcam_ensemble_list[drop_iter]):  # [0][0] stands for getting 8th layer 9th head
                list_of_filtered_token_string = []
                for token_id in txt_tokens_filtered.input_ids[img_num][1:]:
                    word = model_textloc.module.tokenizer.decode([token_id])
                    if token_id == 102:
                        break
                    else:
                        list_of_filtered_token_string.append(word)
                list_of_filtered_token_string = list_of_filtered_token_string[3:]
                special_token_char = '##'
                gradcam_filtered = gradcam_filtered[
                                   3:-1]  # [class tokens(enc delete in BIMT file, 3:-1 delete a picture of and sep token),48,48]
                if len(list_of_filtered_token_string) != len(class_filtered_list[img_num]):
                    filtered_average_attention_map = torch.zeros(
                        (len(class_filtered_list[img_num]), gradcam_filtered.shape[1], gradcam_filtered.shape[2]),
                        device=gradcam_filtered.device)
                    ind_token = 0
                    ind_classes = 0
                    while (ind_token < len(list_of_filtered_token_string)):
                        if not (list_of_filtered_token_string[ind_token].startswith(special_token_char)):
                            # filtered_average_attention_map.append(cam_att[:,ind_token, :, :].detach().clone())
                            filtered_average_attention_map[ind_classes, :, :] = gradcam_filtered[ind_token, :,
                                                                                :].detach().clone()
                            # print("start word:", list_of_filtered_token_string[ind_token], ind_token, ind_classes)
                            if (ind_token + 1 < len(list_of_filtered_token_string) and not (
                                    list_of_filtered_token_string[ind_token + 1].startswith(
                                        special_token_char))):  # 如果不要这个的话就是只取split word 的第一个词
                                ind_classes += 1
                            ind_token += 1
                            word_length = 1
                        elif (list_of_filtered_token_string[ind_token].startswith(special_token_char)):
                            word_length += 1
                            filtered_average_attention_map[ind_classes, :, :] = filtered_average_attention_map[
                                                                                ind_classes, :,
                                                                                :].detach().clone() + gradcam_filtered[
                                                                                                      ind_token, :,
                                                                                                      :].detach().clone()
                            if (ind_token + 1 < len(list_of_filtered_token_string) and not (
                                    list_of_filtered_token_string[ind_token + 1].startswith(special_token_char))):
                                filtered_average_attention_map[ind_classes, :, :] /= word_length
                                ind_classes += 1
                            # print("##split word:", list_of_filtered_token_string[ind_token])
                            ind_token += 1

                    filtered_average_attention_map_list.append(filtered_average_attention_map)
                else:
                    filtered_average_attention_map_list.append(gradcam_filtered[:len(class_filtered_list[img_num])])

            # for img, img_id in enumerate(img_ids):
            #
            #         for i, union_Att in enumerate(filtered_average_attention_map_list[img]):
            #             Path(
            #                 f"{args.save_path}/Blip_reinferene_Drop_check/drop_iter{drop_iter}/").mkdir(
            #                 parents=True, exist_ok=True)
            #             gradcam_image_c = getAttMap(norm_imgs_for_drop_list[drop_iter][img].cpu().numpy(),
            #                                         union_Att.cpu().numpy(),
            #                                         blur=False)  # range between 0-1
            #             im_gradcam_c = Image.fromarray((gradcam_image_c * 255).astype(np.uint8), 'RGB')
            #             im_gradcam_c.save(
            #                 f"{args.save_path}/Blip_reinferene_Drop_check/drop_iter{drop_iter}/drop_iter{drop_iter}_Class_{class_filtered_list[img][i]}_UnionGradcam_img_{img_id}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.jpeg")

        gradcam_0_filtered = gradcam_ensemble_list[0].detach().clone()
        gradcam_agg_dropfiltered = gradcam_ensemble_list[0].detach().clone()
        for drop_iter in range(args.drop_iter):
            # print(1281, drop_iter, gradcam_agg_dropfiltered.sum())
            gradcam_agg_dropfiltered += gradcam_ensemble_list[drop_iter].detach().clone()
            # print(1283, drop_iter, gradcam_agg_dropfiltered.sum(), gradcam_0_filtered.sum())
    return gradcam_0_filtered, gradcam_agg_dropfiltered



def Load_predicted_classes(args, nms, best_class_idx_list, class_filtered_list, caption_filtered_list, gt_class_name_list, img_ids, img, pred_path):

    ''' GPT4o without boundary - high precision'''
    save_path = './GPT4o_classification'
    if args.data_type == "voc":
        path = f"{save_path}/voc_classification_noboundary.json"
    elif args.data_type == "psc":
        path = f"{save_path}/psc_classification_noboundary.json"
    elif args.data_type == "ade20k":
        path = f"{save_path}/ade20k_classification_noboundary.json"

    with open(path, 'r') as f:
        cls_result = json.load(f)

    if args.data_type == "ade20k":
        per_img_cls = cls_result["ADE_val_" + img_ids[img].rjust(8, '0')]
    else:
        per_img_cls = cls_result[img_ids[img]]

    try:
        per_img_cls_prb_list = per_img_cls.replace(']\n\n[', '], [').replace('],\n\n[', '], [').replace('], \n[', '], [ ').replace('],\n[', '], [ ').strip("][").split("], [")
        # per_img_cls_list = per_img_cls_prb_list[0].split(",") # for running the first attenmpt with extra attention on the boundary objects
        # per_img_cls_idx_list = [int(cls.split(":")[0]) for cls in per_img_cls_list] # for running the first attenmpt with extra attention on the boundary objects

        # for running the 2nd attenmpt wo extra attention on the boundary objects
        per_img_cls_list = per_img_cls_prb_list[0].split(",")
        if len(per_img_cls_prb_list) == 1 and per_img_cls_prb_list[0]=='':
            print("no output")
            per_img_cls_list = ["1: 'wall'" for i in range(len(per_img_cls_list))]
            per_img_prob_list = [100 for i in range(len(per_img_cls_list))]
        # elif len(per_img_cls_prb_list) == 1 and per_img_cls_prb_list[0] !='':
        #     print("no prob output")
        #     per_img_prob_list = [100 for i in range(len(per_img_cls_list))]
        else:

            per_img_prob_list = per_img_cls_prb_list[1].split(",")
            per_img_prob_list = [int(prob.split(":")[-1].split("%")[0]) for prob in per_img_prob_list]
            # print("regular output", per_img_cls_list, per_img_prob_list)
        per_img_cls_idx_list = []
        for i,prob in enumerate(per_img_prob_list):
            if prob > 70:
                # print("regular output", prob, int(per_img_cls_list[i].split(":")[0]))
                per_img_cls_idx_list.append(int(per_img_cls_list[i].split(":")[0]))

    except:
        breakpoint()

    best_class_idx = []
    cls_filtered = []
    for i in per_img_cls_idx_list:
        best_class_idx.append(i-1)
        cls_filtered.append(nms[i-1])

    if len(best_class_idx) == 0:
        best_class_idx.append(0)
        cls_filtered.append(nms[0])
    best_class_idx_list.append(best_class_idx)
    class_filtered_list.append(cls_filtered)
    caption_filtered_list.append("A picture of " + " ".join(cls_filtered))
    print(891, img_ids[img], gt_class_name_list[img], cls_filtered)


    return best_class_idx_list, class_filtered_list, caption_filtered_list


def Draw_Segmentation_map(args, final_2ndround_argmax_map, label_trues, org_img_list, img_ids, img, filename):
    # Draw all classes segmentation map
    from skimage import color
    import matplotlib.pyplot as plt

    np.random.seed(0)
    color_map = np.random.shuffle(np.random.random((183, 3)))
    seg_map = color.label2rgb(final_2ndround_argmax_map, org_img_list[img], bg_label=0, colors=color_map)
    Path(f"{args.save_path}/0519_Segmentation/").mkdir(parents=True, exist_ok=True)

    plt.imsave(
        f"{args.save_path}/0519_Segmentation/{filename}_{img_ids[img]}_{args.postprocess}.jpeg",
        seg_map)

    gt_map = color.label2rgb(label_trues[img], org_img_list[img], bg_label=0,
                             colors=color_map)
    plt.imsave(
        f"{args.save_path}/0519_Segmentation/GT_{img_ids[img]}.jpeg", gt_map)


def Mean_over_filtered_label_tokens(model_textloc, txt_tokens_filtered, gradcam_filtered, class_filtered_list, img_num):
    list_of_filtered_token_string = []
    for token_id in txt_tokens_filtered.input_ids[img_num][1:]:
        word = model_textloc.module.tokenizer.decode([token_id])
        if token_id == 102:
            break
        else:
            list_of_filtered_token_string.append(word)
    list_of_filtered_token_string = list_of_filtered_token_string[3:]
    special_token_char = '##'
    gradcam_filtered = gradcam_filtered[
                       3:-1]  # [class tokens(enc delete in BIMT file, 3:-1 delete a picture of and sep token),48,48]
    if len(list_of_filtered_token_string) != len(class_filtered_list[img_num]):
        filtered_average_attention_map = torch.zeros(
            (len(class_filtered_list[img_num]), gradcam_filtered.shape[1], gradcam_filtered.shape[2]),
            device=gradcam_filtered.device)
        ind_token = 0
        ind_classes = 0
        while (ind_token < len(list_of_filtered_token_string)):
            if not (list_of_filtered_token_string[ind_token].startswith(special_token_char)):
                # filtered_average_attention_map.append(cam_att[:,ind_token, :, :].detach().clone())
                filtered_average_attention_map[ind_classes, :, :] = gradcam_filtered[ind_token, :, :].detach().clone()
                # print("start word:", list_of_filtered_token_string[ind_token], ind_token, ind_classes)
                if (ind_token + 1 < len(list_of_filtered_token_string) and not (
                list_of_filtered_token_string[ind_token + 1].startswith(
                        special_token_char))):  # 如果不要这个的话就是只取split word 的第一个词
                    ind_classes += 1
                ind_token += 1
                word_length = 1
            elif (list_of_filtered_token_string[ind_token].startswith(special_token_char)):
                word_length += 1
                filtered_average_attention_map[ind_classes, :, :] = filtered_average_attention_map[ind_classes, :,
                                                                    :].detach().clone() + gradcam_filtered[ind_token, :,
                                                                                          :].detach().clone()
                if (ind_token + 1 < len(list_of_filtered_token_string) and not (
                list_of_filtered_token_string[ind_token + 1].startswith(special_token_char))):
                    filtered_average_attention_map[ind_classes, :, :] /= word_length
                    ind_classes += 1
                # print("##split word:", list_of_filtered_token_string[ind_token])
                ind_token += 1

        return filtered_average_attention_map
    else:
        return gradcam_filtered[:len(class_filtered_list[img_num])]


def Mean_over_full_label_tokens(model_textloc, txt_tokens, cam_att, nms):
    ''' Agg label with multiple token to get the final C*P*P Gradcam Tensor'''

    list_of_token_string = []
    for token_id in txt_tokens.input_ids[0][1:]:
        word = model_textloc.module.tokenizer.decode([token_id])
        if token_id == 102:
            break
        else:
            list_of_token_string.append(word)
    list_of_token_string = list_of_token_string[3:]
    special_token_char = '##'
    cam_att = cam_att[:, 3:-1]  # [2,269(without [enc] token and [sep] token),48,48]
    if len(list_of_token_string) != len(nms):
        new_average_attention_map = torch.zeros((cam_att.shape[0], len(nms), cam_att.shape[2], cam_att.shape[3]),
                                                device=cam_att.device)
        ind_token = 0
        ind_classes = 0
        while (ind_token < len(list_of_token_string)):
            if not (list_of_token_string[ind_token].startswith(special_token_char)):
                # new_average_attention_map.append(cam_att[:,ind_token, :, :].detach().clone())
                new_average_attention_map[:, ind_classes, :, :] = cam_att[:, ind_token, :, :].detach().clone()
                # print("start word:", list_of_token_string[ind_token], ind_token, ind_classes)
                if (ind_token + 1 < len(list_of_token_string) and not (
                list_of_token_string[ind_token + 1].startswith(
                        special_token_char))):  # 如果不要这个的话就是只取split word 的第一个词
                    ind_classes += 1
                ind_token += 1
                word_length = 1
            elif (list_of_token_string[ind_token].startswith(special_token_char)):
                word_length += 1
                new_average_attention_map[:, ind_classes, :, :] = new_average_attention_map[:, ind_classes, :,
                                                                  :].detach().clone() + cam_att[:, ind_token, :,
                                                                                        :].detach().clone()
                if (ind_token + 1 < len(list_of_token_string) and not (
                list_of_token_string[ind_token + 1].startswith(special_token_char))):
                    new_average_attention_map[:, ind_classes, :, :] /= word_length
                    ind_classes += 1
                # print("##split word:", list_of_token_string[ind_token])
                ind_token += 1

    return new_average_attention_map



def Load_GroundTruth(args, img_ids):
    '''Load Ground Truth mask for Evaluation'''
    if args.data_type == "voc":
        gt_path = './VOCdevkit/VOC2012/SegmentationClass'
        label_trues = []
        for img_id in img_ids:
            mask = np.float32(Image.open(os.path.join(gt_path, img_id + '.png')))
            mask[mask == 255] = 0
            label_trues.append(mask)
    elif args.data_type == "psc":
        gt_path = './mmsegmentation/data/VOCdevkit/VOC2010/SegmentationClassContext/'
        label_trues = []
        for img_id in img_ids:
            mask = np.float32(Image.open(os.path.join(gt_path, img_id + '.png')))
            # mask[mask == 255] = 0  #GT 0 is background
            label_trues.append(mask)
    elif args.data_type == "ade20k":
        label_trues = []
        for img_id in img_ids:
            img_id = "ADE_val_" + img_id.rjust(8, '0')
            mask = np.float32(
                Image.open(os.path.join('./ADEChallengeData2016/annotations/validation/', img_id + '.png')))
            # mask[mask == 255] = 0 # background is 0 already
            label_trues.append(mask)


    return label_trues

def load_OrgImage(args, img_ids):
    '''Load OG image for denseCRF'''
    if args.data_type == "voc":
        orgimg_path = "./VOCdevkit/VOC2012/JPEGImages/"
        org_img_list = []
        for img_id in img_ids:
            org_image = Image.open(os.path.join(orgimg_path, img_id + '.jpg')).convert('RGB')
            org_image = np.asarray(org_image)
            org_img_list.append(org_image)
    elif args.data_type == "psc":
        orgimg_path = "./VOCdevkit/VOC2012/JPEGImages/"
        org_img_list = []
        for img_id in img_ids:
            org_image = Image.open(os.path.join(orgimg_path, img_id + '.jpg')).convert('RGB')
            org_image = np.asarray(org_image)
            org_img_list.append(org_image)
    elif args.data_type == "ade20k":
        org_img_list = []
        for img_id in img_ids:
            img_id = "ADE_val_" + img_id.rjust(8, '0')
            org_image = Image.open(os.path.join("./ADEChallengeData2016/images/validation/", img_id + '.jpg')).convert(
                'RGB')
            org_image = np.asarray(org_image)
            org_img_list.append(org_image)


    return org_img_list


def Record_classification_prediction(args, nms, cls_type, best_class_idx, caption_filtered, class_filtered, img_ids, img, text_clip_in, gt_class_name_list, full_list_and_drop):
    record_filtered_classes = []
    # record_filtered_idx = []
    for i in best_class_idx:
        record_filtered_classes.append(text_clip_in[i])
    if cls_type == "CLIP largerthantop2mean":
        caption_filtered.append("A picture of " + " ".join(record_filtered_classes))
        class_filtered.append(record_filtered_classes)
    print(766, cls_type, img_ids[img], "record_filtered_classes", record_filtered_classes, "gt classes", gt_class_name_list[img])

    # record the classification preds and classification true for further evaluation
    idx_list = [full_list_and_drop[i] for i in best_class_idx]
    gt_idx = [nms.index(''.join(''.join(classname.split(" ")).split("-"))) for classname in gt_class_name_list[img]]
    classification_pred = np.array([1 if i in idx_list else 0 for i in range(len(nms))])
    classification_true = np.array([1 if i in gt_idx else 0 for i in range(len(nms))])

    Path(
        f"{args.save_path}/classification_pred_{cls_type}/").mkdir(
        parents=True, exist_ok=True)
    Path(
        f"{args.save_path}/classification_true_{cls_type}/").mkdir(
        parents=True, exist_ok=True)

    np.save(
        f"{args.save_path}/classification_pred_{cls_type}/classification_pred_img_{img_ids[img]}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy",
        classification_pred)
    np.save(f"{args.save_path}/classification_true_{cls_type}/img_{img_ids[img]}_max_blocknum_{args.max_att_block_num}_atthead_{args.prune_att_head}.npy",
            classification_true)
    return caption_filtered, class_filtered


def SquarePad(image):
        s = image.size()
        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        hp2 = (max_wh - s[-1]) - hp
        vp = int((max_wh - s[-2]) / 2)
        vp2 = (max_wh - s[-2]) - vp
        padding = (hp, hp2, vp, vp2)
        return F.pad(input=image, pad = padding, value=0, mode='constant')



# #### Blur + CRF
def postprocess(args, final_pred_wbackground, org_img_list, label_trues, img):
    if "blur" in args.postprocess and "crf" in args.postprocess:
        print("blurring+crfing")
        final_pred_wbackground_blur_list = []
        for i in range(final_pred_wbackground.shape[0]):
            final_pred_wbackground_blur = torch.from_numpy(
                blurring(final_pred_wbackground[i], (label_trues[img].shape[0], label_trues[img].shape[1]),
                         scale=0.05))
            final_pred_wbackground_blur_list.append(final_pred_wbackground_blur)
        final_pred_wbackground = torch.stack(final_pred_wbackground_blur_list, axis=0)
        final_pred_argmax_map = densecrf(org_img_list[img], final_pred_wbackground)
    elif "crf" in args.postprocess and not "blur" in args.postprocess:
        print("crfing")
        final_pred_argmax_map = densecrf(org_img_list[img], final_pred_wbackground)
    elif "blur" in args.postprocess and not "crf" in args.postprocess:
        print("blurring")
        final_pred_wbackground_blur_list = []
        for i in range(final_pred_wbackground.shape[0]):
            final_pred_wbackground_blur = torch.from_numpy(
                blurring(final_pred_wbackground[i],
                         (label_trues[img].shape[0], label_trues[img].shape[1]), scale=0.05))
            final_pred_wbackground_blur_list.append(final_pred_wbackground_blur)
        final_pred_wbackground = torch.stack(final_pred_wbackground_blur_list,
                                                       axis=0)
        final_pred_argmax_map = (torch.argmax(final_pred_wbackground, dim=0)).numpy()

    return final_pred_argmax_map

def densecrf(image, mask):
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as utils
    import torch.nn.functional as F
    import torchvision.transforms.functional as VF

    MAX_ITER = 10
    POS_W = 7
    POS_XY_STD = 3
    Bi_W = 10
    Bi_XY_STD = 50 #org50
    Bi_RGB_STD = 5


    # h, w = mask.shape
    # mask = mask.reshape(1, h, w)
    # fg = mask.astype(float)
    # bg = 1 - fg
    # output_logits = torch.from_numpy(np.concatenate((bg, fg), axis=0))
    #
    # H, W = image.shape[:2]
    image = np.ascontiguousarray(image).copy()
    #
    # output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear").squeeze()
    # output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    output_logits = mask
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    MAP = np.argmax(Q, axis=0).reshape((h, w)).astype(np.float32)
    return MAP



def Scale_0_1(AA):
    if len(AA.shape) == 2:
        return AA
    elif len(AA.shape) == 3:
        c, h,w = AA.shape
        AA = AA.view(AA.size(0), -1)
        AA -= AA.min(-1, keepdim=True)[0]
        AA /= AA.max(-1, keepdim=True)[0]
        AA = AA.view(c, h, w)
    elif len(AA.shape) == 4:
        b, c,h,w = AA.shape
        AA = AA.view(AA.size(0),AA.size(1), -1)
        AA -= AA.min(-1, keepdim=True)[0]
        AA /= AA.max(-1, keepdim=True)[0]
        AA = AA.view(b, c, h, w)

    return AA


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


def scores(label_trues, label_preds, cats, n_class):
    hist = np.zeros((n_class, n_class))
    # breakpoint()
    img_count = 0
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
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
        if class_id == 0:
            class_name_list.append("Background")
        else:
            class_name_list.append(cats[int(class_id)])
    # print("69, iu.shape", iu.shape, class_name_list)
    cls_iu = dict(zip(class_name_list, iu))
    # cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }, hist


def blurring( att_resize, img_shape, scale=0.05):
    att_resize = filters.gaussian_filter(att_resize, scale*max(img_shape))
    att_resize = att_resize - att_resize.min()
    att_resize = att_resize / att_resize.max()
    return att_resize

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

    """### Load images into batches"""
    if args.data_type == "voc":
        # imageDir = f"{args.home_dir}/mmsegmentation/data/VOCdevkit/VOC2012"
        imageDir = f"{args.home_dir}/VOCdevkit/VOC2012"
        cats, nms, data_loader_test = load_voc(args, rank, imageDir)
    elif args.data_type == "psc":
        imageDir = f"{args.home_dir}/mmsegmentation/data/VOCdevkit/VOC2010"
        cats, nms, data_loader_test = load_psc(args, rank, imageDir)
    elif args.data_type == "ade20k":
        cats, nms, data_loader_test = load_ade20k(args, rank, imageDir=None)

    """#### Load text localization model and preprocessors"""

    model_textloc, vis_processors_textloc, text_processors_textloc = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=rank, is_eval=True)




    model_textloc = DDP(model_textloc, device_ids=[rank])

    captions_text_loc(args,  nms, model_textloc, data_loader_test,
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
    print(1390, args)
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
        # print("1431 No gt class for the image", batch_text_input_clip)
        # if batch_id > 1000:
        #     break
        # print("running ", img_ids)
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


    # model_textloc = DDP(model_textloc, device_ids=[rank])
    # model_clip = DDP(model_clip, device_ids=[rank])

    if args.search:
        # torch.cuda.set_device(rank)

        # dataDir = f'{args.home_dir}/coco'

        # dataDir = '/home/letitiabanana/BLIP-main/coco/'
        dataType_stuff = 'stuff_val2017'
        annFile_stuff = f'{args.home_dir}/coco/annotations/{dataType_stuff}.json'
        dataType_thing = 'val2017'
        annFile_thing = f'{args.home_dir}/coco/annotations/instances_{dataType_thing}.json'

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
        cats = cats_thing  + cats_stuff
        nms = nms_thing  + nms_stuff
        print("1492", type(coco_thing), type(cats_thing), type(nms_thing))
        print("1501  len(cats), len(nms)", cats, len(nms))
        # breakpoint()

        """### Load coco images into batches"""

        """#### Load text localization model and preprocessors"""

        # model_textloc, vis_processors_textloc, text_processors_textloc = load_model_and_preprocess(
        #     "blip_image_text_matching", "large", is_eval=True)

        model_clip, vis_processors_clip, txt_processors_clip = load_model_and_preprocess("clip_feature_extractor",
                                                                                         model_type="ViT-L-14-336",
                                                                                         is_eval=True)
        # print("1525 check model structure", model_textloc)
        # breakpoint()
        imageDir = f'{args.home_dir}/coco/images/val2017'

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

        # "layer": np.arange(8, 9),
        # "head": np.arange(9, 10),
        # "min_att": np.arange(0.05, 0.1, 0.05),


        # opt = BayesianOptimizer(
        #     search_space,
        #     sampling={"random": 100000},
        # )
        # opt.search(Search, n_iter=500)

        opt = RandomAnnealingOptimizer(search_space, random_state=1164005944)
        opt.search(Search, n_iter=34)

        # opt = RandomSearchOptimizer(search_space, random_state=1164005944)
        # opt.search(Search, n_iter=34)

        # opt = GridSearchOptimizer(
        #     search_space,
        # )
        # # print("1508", opt)
        # opt.search(Search, n_iter=192)

    elif args.cal_token_sim_forall_layerhead:
        # device = "cuda:3"
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
        '''read all the token contrast sum and calculate average for each layer-head'''
         #torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # "cuda:3"
        sum_token_contrast_dict = {}
        for layer in range(1,13):
            for head in range(12):
                sum_std_dict_path = f"Token_Contrast/max_att_block_num{layer}_atthead{head}/Token_contrast_sum/"
                exist_token_contrast_list = []
                for (dirpath, dirnames, filenames) in walk(sum_std_dict_path):
                    exist_token_contrast_list.extend(filenames)
                    break
                token_contrast_sum = 0
                for filename in exist_token_contrast_list:
                    img_id = filename.split("_")[1].split(".")[0]
                    with open(os.path.join(sum_std_dict_path, filename), "r") as outfile:
                        sum_std_dict = json.load(outfile)
                    token_contrast_sum += sum_std_dict[f"TCSum_layer{layer}_head{head}_img_{img_id}"] / len(exist_token_contrast_list)
                sum_token_contrast_dict[f"Total_TCSum_layer{layer}_head{head}"] = token_contrast_sum
        with open(f"Token_Contrast/sum_token_contrast.json", "w") as outfile:
            json.dump(sum_token_contrast_dict, outfile, cls=NumpyArrayEncoder)

    else:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
    ### gather all the att loss record from each gpu