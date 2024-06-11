import torch
from Dataset import PascalVOC, PascalContext, ADE20K, CocoDetection
from torch.utils.data.distributed import DistributedSampler
from argparse import Namespace
from pycocotools.coco import COCO

def load_voc(args, rank, imageDir):
    cats = {1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair",
            10: "cow", 11: "table", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
            16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

    nms = [i for i in cats.values()]


    test_data = PascalVOC(imageDir, split="val", args=args,
                          device=rank)  ##add arguments args, annFile_thing, getClassName, cats, vis_processors_clip

    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        sampler=DistributedSampler(test_data)
    )
    return cats, nms, data_loader_test


def load_psc(args, rank, imageDir):
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
    test_data = PascalContext(imageDir, split="val", args=args,
                              device=rank)  ##add arguments args, annFile_thing, getClassName, cats, vis_processors_clip

    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        sampler=DistributedSampler(test_data)
    )
    return cats, nms, data_loader_test


def load_ade20k(args, rank, imageDir):
    cats = {1:'wall', 2: 'building', 3: 'sky', 4: 'floor', 5: 'tree', 6: 'ceiling', 7: 'road', 8: 'bed',
            9: 'windowpane', 10: 'grass', 11: 'cabinet', 12: 'sidewalk', 13: 'person', 14: 'ground', 15: 'door',
            16: 'table', 17: 'mountain', 18: 'plant', 19: 'curtain', 20: 'chair', 21: 'car', 22: 'water',
            23: 'painting', 24: 'sofa', 25: 'shelf', 26: 'house', 27: 'sea', 28: 'mirror', 29: 'rug',
            30: 'field', 31: 'armchair', 32: 'seat', 33: 'fence', 34: 'desk', 35: 'rock', 36: 'wardrobe',
            37: 'lamp', 38: 'bathtub', 39: 'railing', 40: 'cushion', 41: 'base', 42: 'box', 43: 'pillar',
            44: 'signboard', 45: 'chest of drawers', 46: 'counter', 47: 'sand', 48: 'sink', 49: 'skyscraper',
            50: 'fireplace', 51: 'refrigerator', 52: 'grandstand', 53: 'path', 54: 'stairs', 55: 'runway',
            56: 'case', 57: 'billiard table', 58: 'pillow', 59:  'screen', 60:  'stairway', 61:  'river',
            62:  'bridge', 63:  'bookcase', 64:  'blind', 65:  'coffee table', 66:  'toilet', 67:  'flower',
            68:  'book', 69:  'hill', 70: 'bench', 71: 'countertop', 72: 'stove', 73: 'palm', 74: 'kitchen island',
            75: 'computer', 76: 'swivel chair', 77: 'boat', 78: 'bar', 79: 'arcade machine', 80: 'hovel', 81: 'bus',
            82: 'towel', 83: 'light', 84: 'truck', 85: 'tower', 86: 'chandelier', 87: 'sunshade', 88: 'streetlight',
            89: 'booth', 90: 'television receiver', 91: 'airplane', 92: 'dirt track', 93: 'apparel', 94: 'pole',
            95: 'land', 96: 'bannister', 97: 'escalator', 98: 'ottoman', 99: 'bottle', 100:'buffet', 101:'poster',
            102:'stage', 103:'van', 104:'ship', 105:'fountain', 106:'conveyer belt', 107:'canopy', 108:'washer',
            109:'toy', 110:'swimming pool', 111:'stool', 112:'barrel', 113:'basket', 114:'waterfall', 115:'tent',
            116:'bag', 117:'motorbike', 118:'cradle', 119:'oven', 120:'ball', 121:'food', 122:'stair', 123:'tank',
            124:'marque', 125:'microwave', 126:'pot', 127:'animal', 128:'bicycle', 129:'lake', 130:'dishwasher',
            131:'screen', 132:'blanket', 133:'sculpture', 134:'hood', 135:'sconce', 136:'vase', 137:'trafficlight',
            138:'tray', 139:'trash can', 140:'fan', 141:'pier', 142:'crt screen', 143:'plate', 144:'monitor',
            145:'bulletinboard', 146:'shower', 147:'radiator', 148:'glass', 149:'clock', 150:'flag'}



    nms = ["".join(i.split(" ")) for i in cats.values()]
    root_dataset = f"{args.home_dir}"
    list_val = f"{args.home_dir}/semantic-segmentation-pytorch-master/data/validation.odgt"
    dataset_config = Namespace(num_class=150, padding_constant=32, img_size=args.img_size)

    test_data = ADE20K(root_dataset, list_val, dataset_config)

    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        sampler=DistributedSampler(test_data)
    )

    return cats, nms, data_loader_test


def load_coco(args, rank, imageDir, annFile_thing, cats):




    test_data = CocoDetection(imageDir, args, annFile_thing, getClassName, cats,
                              vis_processors_clip_large=None,
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
    return  data_loader_test


def getClassName(class_id, cats):
    for i in range(len(cats)):
        # print("234 get classname", cats[i]['id']==class_id, type(cats[i]['id']), type(class_id))
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    return "None"
