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
# import mxnet.ndarray as F
# from mxnet import cpu
from os import walk

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


import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        # CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        # CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        # CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        # CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        # CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        # CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        # CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, args, root, split='train', target_type='semantic'):
        self.args = args
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        # self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        img0 = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.targets[index])

        img_id = self.images[index].split(".")[1].split("/")[-1]

        norm_mask = np.float32(mask)
        label = np.zeros(30)
        distinct_label = []

        for ix in np.unique(norm_mask):
            if ix > 6:
                distinct_label.append(ix)
            else:
                continue
        label[:len(distinct_label)] = distinct_label
        path = [1]

        img_size = img0.size

        resized_img = img0.resize((self.args.img_size, self.args.img_size))
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

        return img_size, norm_img, img0, path, img_id, label  # , gt_class_name  #, target, img

        # if self.transform:
        #     image, target = self.transform(image, target)
        # target = self.encode_target(target)
        # return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)


class PascalContext(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, split, args,  vis_processors_clip=None, transform=None, target_transform=None, device="cpu"): #annFile, getClassName, cats,

        self._img_dir = os.path.join(root, 'JPEGImages')
        # .txt split file
        if split == 'train':
            _split_f = os.path.join(root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        # 59 + background labels directory
        _mask_dir = os.path.join(root, 'SegmentationClassContext')

        self.images = []
        self.masks = []
        self.ids = []
        val_file = []
        # for (dirpath, dirnames, filenames) in walk(_mask_dir):
        #     val_file.extend(filenames)
        #     break

        with open(os.path.join("/home/letitiabanana/LAVIS/mmsegmentation/data/VOCdevkit/VOC2010/",
                               "trainval_merged.json"), "r") as outfile:
            trainval = json.load(outfile)
        for image in trainval["images"]:
            if image["phase"] == "val":
                val_file.append(image["file_name"])
        # with open(os.path.join(_split_f), 'r') as lines:
        for line in val_file:
            _image = os.path.join(self._img_dir, line.split(".")[0].strip() + '.jpg')
            assert os.path.isfile(_image)
            self.images.append(_image)
            self.ids.append(line.strip())
            _mask = os.path.join(_mask_dir, line.split(".")[0].strip() + '.png')
            assert os.path.isfile(_mask)
            self.masks.append(_mask)
        assert len(self.images) == len(self.masks) ==len(self.ids)
        self.root = root
        # self.coco = COCO(annFile)
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        # self.getClassName = getClassName
        # self.cats = cats
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
            # if os.path.exists(sum_std_dict_path):
            #     with open(sum_std_dict_path,"r") as outfile:
            #         sum_before = json.load(outfile)
            #     sum_before.update(sum_std_dict)
            #     sum_std_dict_save = sum_before
            # else:
            #     sum_std_dict_save = sum_std_dict
            with open(sum_std_dict_path, "w") as outfile:
                json.dump(sum_std_dict, outfile, cls=NumpyArrayEncoder)


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

        # else:
        #     prediction_mask_dict = combine_drop_iter_withclipsim(args, img_id, cats)
        #     prediction_mask = prediction_mask_dict["att_map"]

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
        img0 = Image.open(self.images[index]).convert('RGB')
        img_id = self.ids[index].split(".")[0]
        mask = Image.open(self.masks[index])
        # F.array(np.array(mask), cpu(0)).astype('int32')

        norm_mask = np.float32(mask)
        label = np.zeros(59)
        distinct_label = np.unique(norm_mask)
        label[:len(distinct_label)] = distinct_label
        path = [1]


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

        return  img_size,  norm_img, img0, path, img_id, label #, gt_class_name  #, target, img


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




