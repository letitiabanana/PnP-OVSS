import openai
from openai import OpenAI
import argparse
import json
import re
import os
import time
from tqdm import tqdm
from Dataset import ADE20K_GPT, PascalVOC_GPT, PascalContext_GPT,, CocoDetection_GPT
import torch
from argparse import Namespace
import base64
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--apikey", type=str, required=True)
# parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
# parser.add_argument("--rewritten_file", type=str, required=True)
parser.add_argument("--data_type", type=str, required=True)



args = parser.parse_args()

client = OpenAI(api_key=args.apikey, timeout=60.0)
def getClassName(class_id, cats):
    for i in range(len(cats)):
        # print("234 get classname", cats[i]['id']==class_id, type(cats[i]['id']), type(class_id))
        if cats[i][0] == class_id:
            return cats[i][1]
    return "None"


print(args.data_type)
args.rewritten_file = f'{args.data_type}_classification_noboundary'

if args.data_type == "coco_object":
    from pycocotools.coco import COCO
    cats = [[ 1,  'person'], [ 2, 'bicycle'],[ 3,  'car'], [ 4,  'motorcycle'],
    [ 5,  'airplane'], [ 6,  'bus'],[ 7,  'train'], [ 8,  'truck'],
    [ 9,  'boat'], [10,  'traffic light'], [ 11,'fire hydrant'], [13,  'stop sign'], [ 14,'parking meter'],
    [15,  'bench'], [16,  'bird'], [17,  'cat'], [18,  'dog'], [19,  'horse'], [20,  'sheep'], [21,  'cow'], [22,'elephant'], [23,'bear'],
            [24,  'zebra'], [25,'giraffe'], [27,  'backpack'], [28,  'umbrella'], [ 31,'handbag'], [ 32,  'tie'], [33,  'suitcase'],
            [ 34,'frisbee'], [ 35,  'skis'], [36,  'snowboard'], [ 37, 'sports ball'], [ 38, 'kite'], [39,  'baseball bat'],
            [ 40, 'baseball glove'], [41,  'skateboard'], [ 42, 'surfboard'], [43, 'tennis racket'], [44,  'bottle'], [46,'wine glass'], [ 47,  'cup'],
            [48,  'fork'], [49, 'knife'], [50, 'spoon'], [51,  'bowl'], [ 52,  'banana'],
    [53,  'apple'], [ 54,  'sandwich'], [55,  'orange'], [ 56,  'broccoli'], [57,  'carrot'], [ 58,  'hot dog'],
            [59,  'pizza'], [ 60,  'donut'], [61,  'cake'], [ 62,  'chair'], [63,  'couch'], [ 64, 'potted plant'],
            [65,  'bed'], [ 67,'dining table'], [ 70, 'toilet'], [72,  'tv'], [ 73, 'laptop'], [ 74,  'mouse'],
            [75,  'remote'], [ 76,'keyboard'], [77,  'cell phone'], [ 78, 'microwave'],
            [79,  'oven'], [ 80, 'toaster'], [81,  'sink'], [82,  'refrigerator'], [ 84,'book'], [ 85,'clock'],
            [ 86,  'vase'], [ 87,'scissors'], [ 88,  'teddy bear'], [89,  'hair drier'], [ 90, 'toothbrush']]

    print(cats)
    nms = [cat[1] for cat in cats]

    imageDir = '/home/letitiabanana/LAVIS/coco/images/val2017'
    dataType_thing = 'val2017'
    annFile_thing = '/home/letitiabanana/LAVIS/coco/annotations/instances_{}.json'.format(dataType_thing)
    coco_thing = COCO(annFile_thing)

    test_data = CocoDetection_GPT(imageDir, args, annFile_thing, cats)  ##add arguments
    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False,
    )
elif args.data_type == "coco_stuff":
    from pycocotools.coco import COCO
    cats =[[ 1,'person'], [ 2, 'bicycle'], [ 3,  'car'], [ 4,  'motorcycle'], [ 5,  'airplane'], [ 6,  'bus'], [ 7,  'train'], [ 8,  'truck'], [ 9,  'boat'], [ 10,  'traffic light'], [ 11,  'fire hydrant'], [ 13,  'stop sign'], [ 14,  'parking meter'], [ 15,  'bench'], [16,  'bird'], [17,  'cat'], [18,  'dog'], [19,  'horse'], [20,  'sheep'], [21,  'cow'], [22,  'elephant'], [23,  'bear'], [24,  'zebra'], [25,  'giraffe'], [ 27,  'backpack'], [ 28,  'umbrella'], [ 31,  'handbag'], [ 32,  'tie'], [ 33,  'suitcase'], [ 34,  'frisbee'], [ 35,  'skis'], [ 36,  'snowboard'], [ 37,  'sports ball'], [ 38,  'kite'], [ 39,  'baseball bat'], [ 40,  'baseball glove'], [ 41,  'skateboard'], [ 42,  'surfboard'], [ 43,  'tennis racket'], [44,  'bottle'], [46,  'wine glass'], [47,  'cup'], [48,  'fork'], [49,  'knife'], [50,  'spoon'], [51,  'bowl'], [ 52,  'banana'], [ 53,  'apple'], [ 54,  'sandwich'], [ 55,  'orange'], [ 56,  'broccoli'], [ 57,  'carrot'], [ 58,  'hot dog'], [ 59,  'pizza'], [ 60,  'donut'], [ 61,  'cake'], [ 62,  'chair'], [ 63,  'couch'], [ 64,  'potted plant'], [ 65,  'bed'], [ 67,  'dining table'], [ 70,  'toilet'], [ 72,  'tv'], [ 73,  'laptop'], [ 74,  'mouse'], [ 75,  'remote'], [ 76,  'keyboard'], [ 77,  'cell phone'], [ 78,  'microwave'], [ 79,  'oven'], [ 80,  'toaster'], [ 81,  'sink'], [ 82,  'refrigerator'], [ 84,  'book'], [ 85,  'clock'], [ 86,  'vase'], [ 87,  'scissors'], [ 88,  'teddy bear'], [ 89,  'hair drier'], [ 90,  'toothbrush'], [ 92,  'banner'], [ 93,  'blanket'], [94,  'branch'], [ 95,  'bridge'], [ 96,  'building-other'], [97,  'bush'], [ 98,  'cabinet'], [99,  'cage'], [ 100,  'cardboard'], [ 101,  'carpet'], [102,  'ceiling-other'], [103,  'ceiling-tile'], [ 104,  'cloth'], [ 105,  'clothes'], [ 106,  'clouds'], [ 107,  'counter'], [ 108,  'cupboard'], [ 109,  'curtain'], [ 110,  'desk-stuff'], [111,  'dirt'], [ 112,  'door-stuff'], [113,  'fence'], [ 114,  'floor-marble'], [ 115,  'floor-other'], [ 116,  'floor-stone'], [ 117,  'floor-tile'], [ 118,  'floor-wood'], [119,  'flower'], [120,  'fog'], [121,  'food-other'], [122,  'fruit'], [ 123,  'furniture-other'], [124,  'grass'], [125,  'gravel'], [126,  'ground-other'], [ 127,  'hill'], [ 128,  'house'], [129,  'leaves'], [ 130,  'light'], [ 131,  'mat'], [ 132,  'metal'], [ 133,  'mirror-stuff'], [134,  'moss'], [ 135,  'mountain'], [136,  'mud'], [ 137,  'napkin'], [138,  'net'], [ 139,  'paper'], [140,  'pavement'], [ 141,  'pillow'], [142,  'plant-other'], [ 143,  'plastic'], [144,  'platform'], [145,  'playingfield'], [146,  'railing'], [147,  'railroad'], [148,  'river'], [149,  'road'], [ 150,  'rock'], [ 151,  'roof'], [ 152,  'rug'], [153,  'salad'], [154,  'sand'], [155,  'sea'], [ 156,  'shelf'], [ 157,  'sky-other'], [ 158,  'skyscraper'], [159,  'snow'], [ 160,  'solid-other'], [ 161,  'stairs'], [ 162,  'stone'], [163,  'straw'], [164,  'structural-other'], [ 165,  'table'], [ 166,  'tent'], [ 167,  'textile-other'], [ 168,  'towel'], [169,  'tree'], [170,  'vegetable'], [ 171,  'wall-brick'], [ 172,  'wall-concrete'], [ 173,  'wall-other'], [ 174,  'wall-panel'], [ 175,  'wall-stone'], [ 176,  'wall-tile'], [ 177,  'wall-wood'], [178,  'water-other'], [179,  'waterdrops'], [ 180,  'window-blind'], [ 181,  'window-other'], [ 182,  'wood']]
    print(cats)
    nms = [cat[1] for cat in cats]
    imageDir = '/home/letitiabanana/LAVIS/coco/images/val2017'
    dataType_thing = 'val2017'
    annFile_thing = '/home/letitiabanana/LAVIS/coco/annotations/instances_{}.json'.format(dataType_thing)
    coco_thing = COCO(annFile_thing)

    test_data = CocoDetection_GPT(imageDir, args, annFile_thing, cats)  ##add arguments
    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False,
    )


elif args.data_type == "psc":

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

    nms = ["".join(i.split(" ")) for i in cats.values()]

    imageDir = "/home/letitiabanana/LAVIS/mmsegmentation/data/VOCdevkit/VOC2010"
    test_data = PascalContext_GPT(imageDir, split="val", args=args)  ##add arguments args, annFile_thing, getClassName, cats, vis_processors_clip

    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False,
    )

elif args.data_type == "voc":
    cats = {1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair",
            10: "cow", 11: "table", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
            16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}
    nms = ["".join(i.split(" ")) for i in cats.values()]

    imageDir = "/home/letitiabanana/LAVIS/VOCdevkit/VOC2012/"
    test_data = PascalVOC_GPT(imageDir, split="val", args=args)  ##add arguments args, annFile_thing, getClassName, cats, vis_processors_clip

    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False,
    )

elif args.data_type == "ade20k":
    cats = {1: 'wall', 2: 'building;edifice', 3: 'sky', 4: 'floor;flooring', 5: 'tree', 6: 'ceiling', 7: 'road;route',
            8: 'bed', 9: 'windowpane;window', 10: 'grass', 11: 'cabinet', 12: 'sidewalk;pavement',
            13: 'person;individual;someone;somebody;mortal;soul', 14: 'earth;ground', 15: 'door;double;door', 16: 'table',
            17: 'mountain;mount', 18: 'plant;flora;plant;life', 19: 'curtain;drape;drapery;mantle;pall', 20: 'chair',
            21: 'car;auto;automobile;machine;motorcar', 22: 'water', 23: 'painting;picture', 24: 'sofa;couch;lounge',
            25: 'shelf', 26: 'house', 27: 'sea', 28: 'mirror', 29: 'rug;carpet;carpeting', 30: 'field', 31: 'armchair',
            32: 'seat', 33: 'fence;fencing', 34: 'desk', 35: 'rock;stone', 36: 'wardrobe;closet;press', 37: 'lamp',
            38: 'bathtub;bathing;tub;bath;tub', 39: 'railing;rail', 40: 'cushion', 41: 'base;pedestal;stand', 42: 'box',
            43: 'column;pillar', 44: 'signboard;sign', 45: 'chest;of;drawers;chest;bureau;dresser', 46: 'counter',
            47: 'sand',
            48: 'sink', 49: 'skyscraper', 50: 'fireplace;hearth;open;fireplace', 51: 'refrigerator;icebox',
            52: 'grandstand;covered;stand', 53: 'path', 54: 'stairs;steps', 55: 'runway',
            56: 'case;display;case;showcase;vitrine', 57: 'pool;table;billiard;table;snooker;table', 58: 'pillow',
            59: 'screen;door;screen', 60: 'stairway;staircase', 61: 'river', 62: 'bridge;span', 63: 'bookcase',
            64: 'blind;screen', 65: 'coffee;table;cocktail;table', 66: 'toilet;can;commode;crapper;pot;potty;stool;throne',
            67: 'flower', 68: 'book', 69: 'hill', 70: 'bench', 71: 'countertop',
            72: 'stove;kitchen;stove;range;kitchen;range;cooking;stove', 73: 'palm;palm;tree', 74: 'kitchen;island',
            75: 'computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system',
            76: 'swivel;chair', 77: 'boat', 78: 'bar', 79: 'arcade;machine', 80: 'hovel;hut;hutch;shack;shanty',
            81: 'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle',
            82: 'towel',
            83: 'light;light;source', 84: 'truck;motortruck', 85: 'tower', 86: 'chandelier;pendant;pendent',
            87: 'awning;sunshade;sunblind', 88: 'streetlight;street;lamp', 89: 'booth;cubicle;stall;kiosk',
            90: 'television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box',
            91: 'airplane;aeroplane;plane', 92: 'dirt;track', 93: 'apparel;wearing;apparel;dress;clothes', 94: 'pole',
            95: 'land;ground;soil', 96: 'bannister;banister;balustrade;balusters;handrail',
            97: 'escalator;moving;staircase;moving;stairway', 98: 'ottoman;pouf;pouffe;puff;hassock', 99: 'bottle',
            100: 'buffet;counter;sideboard', 101: 'poster;posting;placard;notice;bill;card', 102: 'stage', 103: 'van',
            104: 'ship', 105: 'fountain', 106: 'conveyer;belt;conveyor;belt;conveyer;conveyor;transporter', 107: 'canopy',
            108: 'washer;automatic;washer;washing;machine', 109: 'plaything;toy',
            110: 'swimming;pool;swimming;bath;natatorium', 111: 'stool', 112: 'barrel;cask', 113: 'basket;handbasket',
            114: 'waterfall;falls', 115: 'tent;collapsible;shelter', 116: 'bag', 117: 'minibike;motorbike', 118: 'cradle',
            119: 'oven', 120: 'ball', 121: 'food;solid;food', 122: 'step;stair', 123: 'tank;storage;tank',
            124: 'trade;name;brand;name;brand;marque', 125: 'microwave;microwave;oven', 126: 'pot;flowerpot',
            127: 'animal;animate;being;beast;brute;creature;fauna', 128: 'bicycle;bike;wheel;cycle', 129: 'lake',
            130: 'dishwasher;dish;washer;dishwashing;machine', 131: 'screen;silver;screen;projection;screen',
            132: 'blanket;cover', 133: 'sculpture', 134: 'hood;exhaust;hood', 135: 'sconce', 136: 'vase',
            137: 'traffic;light;traffic;signal;stoplight', 138: 'tray',
            139: 'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin', 140: 'fan',
            141: 'pier;wharf;wharfage;dock', 142: 'crt;screen', 143: 'plate', 144: 'monitor;monitoring;device',
            145: 'bulletin;board;notice;board', 146: 'shower', 147: 'radiator', 148: 'glass;drinking;glass', 149: 'clock',
            150: 'flag'}


    nms = ["".join(i.split(" ")) for i in cats.values()]

    root_dataset = "./"
    list_val = "./semantic-segmentation-pytorch-master/data/validation.odgt"
    dataset_config = Namespace(num_class=150, padding_constant=32)

    test_data = ADE20K_GPT(root_dataset, list_val, dataset_config)

    torch.manual_seed(10000)
    data_loader_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False
    )



save_path = os.path.join(args.output_dir, args.rewritten_file + '.json')
if os.path.exists(save_path):
    with open(save_path, 'r') as f:
        exist = json.load(f)
    start = len(exist)
else:
    start = 0
    exist = {}

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
gens = {}
for idx, (img_path, segm_path) in tqdm(enumerate(data_loader_test)):
    img_id = segm_path[0].split("/")[-1].split(".")[0]
    if img_id not in exist.keys() or exist[img_id] =='':
        print(217, img_id)
    # if idx >= start and idx <=3:
        if args.data_type == "coco_object":
            img_id = segm_path[0].split("/")[-1].split(".")[0]
            id_coco = int(img_id.lstrip("0"))
            ann_ids_thing = coco_thing.getAnnIds(imgIds=id_coco)
            target = coco_thing.loadAnns(ann_ids_thing)
            gt_class_name = []
            for i in range(len(target)):
                if target[i]["category_id"] == 183:
                    continue
                gt_class = getClassName(target[i]["category_id"], cats)
                # print("104 gt cls", target[i]["category_id"], gt_class)
                if gt_class not in gt_class_name:
                    gt_class_name.append(gt_class)
        elif args.data_type == "coco_stuff":
            img_id = segm_path[0].split("/")[-1].split(".")[0]
            segm = np.float32(Image.open(segm_path[0])) #0-182，255
            gt_idx = np.unique(segm)+1
            gt_class_name = []
            for i in gt_idx:
                if int(i) != 255 and int(i) != 183:
                    gt_class_name.append(getClassName(i, cats))

        else:
            img_id = segm_path[0].split("/")[-1].split(".")[0]
            segm = np.float32(Image.open(segm_path[0]))
            gt_idx = np.unique(segm)
            gt_class_name = []
            for i in gt_idx:
                if i != 0 and i != 255:
                    gt_class_name.append(cats[i])

        base64_image = encode_image(img_path[0])
        attempt = 0
        no_timeout = False
        while attempt < 3:
            try:
                response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": f"Help me find as much and accurate as possible, categoreis appear in the image among the availble categories {cats}. Double check if there is anything missing. You may output unsure categories and give them lower probability, but at most 25 categories "
                                                     f"Ouput strictly in the format [id1: classname1, id2: classname2, ...], [probability of class1 in percentage, probability of class2 in percentage, ...] without other words. you have to at least ouput one cats. "},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"}
                             }
                        ]}
                    ],
                    temperature=0.0,
                )
                gen_content = response.choices[0].message.content

                gens[img_id] = gen_content
                attempt = 3
                no_timeout = True
                print(293, img_id, gen_content, gt_class_name)
            except openai.BadRequestError:
                print("BadRequestError")
                attempt = 3
                gens[img_id] = ''
                error_file = os.path.join(args.output_dir, 'no_output_idx.json')
                if os.path.exists(error_file):
                    with open(error_file, 'r') as f:
                        bad_idx_list = json.load(f)
                    bad_idx_list.append(img_id)
                else:
                    bad_idx_list = [img_id]

                with open(error_file, 'w') as f:
                    json.dump(bad_idx_list, f)

                no_timeout = True
            except:
                print("timeout")
                time.sleep(20)
                attempt += 1
            try:
                print(293, img_id,  gen_content, gt_class_name)
            except:
                error_file = os.path.join(args.output_dir, 'no_return.json')
                if os.path.exists(error_file):
                    with open(error_file, 'r') as f:
                        bad_idx_list = json.load(f)
                    bad_idx_list.append(img_id)
                else:
                    bad_idx_list = [img_id]

                with open(error_file, 'w') as f:
                    json.dump(bad_idx_list, f)
        if no_timeout is False:
            gens[img_id] = ''
            error_file = os.path.join(args.output_dir, 'no_output_idx.json')
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    bad_idx_list = json.load(f)
                bad_idx_list.append(img_id)
            else:
                bad_idx_list = [img_id]

            with open(error_file, 'w') as f:
                json.dump(bad_idx_list, f)


        # if len(gens) % 2 == 0:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if not os.path.exists(save_path):
            with open(save_path, 'w') as f:
                json.dump(gens, f)
        else:
            with open(save_path, 'r') as f:
                tmp = json.load(f)
            tmp.update(gens)

            with open(save_path, 'w') as f:
                json.dump(tmp, f)

        gens = {}
