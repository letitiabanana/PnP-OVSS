import os
import numpy as np
import mxnet as mx
from PIL import Image
from tqdm import trange
from gluoncv.data.segbase import SegmentationDataset
from mxnetseg.utils import DATASETS, dataset_dir


# @DATASETS.add_component
# class PascalContext(SegmentationDataset):
#     """
#     Pascal context dataset.
#     Reference: R. Mottaghi, et al. The role of context for object detection and semantic
#         segmentation in the wild. CVPR 2014.
#     """
#     NUM_CLASS = 59
#
#     def __init__(self, root=None, split='val', mode=None, transform=None, **kwargs):
#         root = root if root is not None else os.path.join(dataset_dir(), 'PContext')
#         super(PascalContext, self).__init__(root, split, mode, transform, **kwargs)
#         self._img_dir = os.path.join(root, 'JPEGImages')
#         # .txt split file
#         if split == 'train':
#             _split_f = os.path.join(root, 'train.txt')
#         elif split == 'val':
#             _split_f = os.path.join(root, 'val.txt')
#         else:
#             raise RuntimeError('Unknown dataset split: {}'.format(split))
#         if not os.path.exists(_split_f):
#             self._generate_split_f(_split_f)
#         # 59 + background labels directory
#         _mask_dir = os.path.join(root, 'Labels_59')
#         if not os.path.exists(_mask_dir):
#             self._preprocess_mask(_mask_dir)
#
#         self.images = []
#         self.masks = []
#         with open(os.path.join(_split_f), 'r') as lines:
#             for line in lines:
#                 _image = os.path.join(self._img_dir, line.strip() + '.jpg')
#                 assert os.path.isfile(_image)
#                 self.images.append(_image)
#
#                 _mask = os.path.join(_mask_dir, line.strip() + '.png')
#                 assert os.path.isfile(_mask)
#                 self.masks.append(_mask)
#         assert len(self.images) == len(self.masks)
#
#     def _get_imgs(self, split='trainval'):
#         """ get images by split type using Detail API. """
#         from detail import Detail
#         annotation = os.path.join(self.root, 'trainval_merged.json')
#         detail = Detail(annotation, self._img_dir, split)
#         imgs = detail.getImgs()
#         return imgs, detail
#
#     def _generate_split_f(self, split_f):
#         print("Processing %s...Only run once to generate this split file." % (self.split + '.txt'))
#         imgs, _ = self._get_imgs(self.split)
#         img_list = []
#         for img in imgs:
#             file_id, _ = img.get('file_name').split('.')
#             img_list.append(file_id)
#         with open(split_f, 'a') as split_file:
#             split_file.write('\n'.join(img_list))
#
#     @staticmethod
#     def _class_to_index(mapping, key, mask):
#         # assert the values
#         values = np.unique(mask)
#         for i, values in enumerate(values):
#             assert (values in mapping)
#         index = np.digitize(mask.ravel(), mapping, right=True)
#         return key[index].reshape(mask.shape)
#
#     def _preprocess_mask(self, _mask_dir):
#         print("Processing mask...Only run once to generate 59-class mask.")
#         os.makedirs(_mask_dir)
#         mapping = np.sort(np.array([
#             0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
#             23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
#             427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
#             68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
#             98, 187, 104, 105, 366, 189, 368, 113, 115]))
#         key = np.array(range(len(mapping))).astype('uint8')
#         imgs, detail = self._get_imgs()
#         bar = trange(len(imgs))
#         for i in bar:
#             img = imgs[i]
#             img_name, _ = img.get('file_name').split('.')
#             mask = Image.fromarray(self._class_to_index(mapping, key, detail.getMask(img)))
#             mask.save(os.path.join(_mask_dir, img_name + '.png'))
#             bar.set_description("Processing mask {}".format(img.get('image_id')))
#
#     def __getitem__(self, idx):
#         img = Image.open(self.images[idx]).convert('RGB')
#         mask = Image.open(self.masks[idx])
#         # synchronized transform
#         if self.mode == 'train':
#             img, mask = self._sync_transform(img, mask)
#         elif self.mode == 'val':
#             img, mask = self._val_sync_transform(img, mask)
#         else:
#             assert self.mode == 'testval'
#             img, mask = self._img_transform(img), self._mask_transform(mask)
#         # general resize, normalize and toTensor
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, mask
#
#     def _mask_transform(self, mask):
#         target = np.array(mask).astype('int32') - 1  # ignore background
#         return mx.nd.array(target, mx.cpu(0))
#
#     def __len__(self):
#         return len(self.images)
#
#     @property
#     def classes(self):
#         # background is ignored
#         return ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#                 'chair', 'cow', 'table', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
#                 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed', 'bench', 'book', 'building',
#                 'cabinet', 'ceiling', 'cloth', 'computer', 'cup', 'door', 'fence', 'floor', 'flower',
#                 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain',
#                 'platform', 'sign', 'plate', 'road', 'rock', 'shelves', 'sidewalk', 'sky',
#                 'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood')

#
import cv2
import numpy as np
from torchvision.datasets import VOCSegmentation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class PascalVOCSearchDataset(VOCSegmentation):
    def __init__(self, args, root="~/data/pascal_voc", image_set="val", download=True, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        img0 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        # if self.transform is not None:
        #     transformed = self.transform(image=image, mask=mask)
        #     image = transformed["image"]
        #     mask = transformed["mask"]

        img_size = img0.size

        resized_img = img0.resize((768, 768))
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
        return img0, mask
