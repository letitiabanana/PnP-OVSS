import numpy as np
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Calculate mIoU', add_help=False)
    parser.add_argument('--data_type', default="coco object", type=str)
    parser.add_argument('--save_path', default=None, type=str)
    return parser


def getClassName(class_id, cats):
    if class_id == 0:
        return "Background"
    for i in range(len(cats)):
        if cats[i]['id'] == class_id:
            return cats[i]['name']
    print("Not a coco class")
    return "None class"


def main():
    if args.data_type == "coco_object":
        cats = [{'supercategory': 'person', 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}, {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]
    elif args.data_type == "coco_stuff":
        cats = [{'supercategory': 'person', 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
         {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
         {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
         {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
         {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
         {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
         {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
         {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
         {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
         {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
         {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
         {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
         {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
         {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
         {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
         {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
         {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
         {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
         {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
         {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
         {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
         {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
         {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
         {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
         {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
         {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
         {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
         {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
         {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
         {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
         {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
         {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
         {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
         {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'id': 52, 'name': 'banana'},
         {'supercategory': 'food', 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
         {'supercategory': 'food', 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
         {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
         {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'id': 60, 'name': 'donut'},
         {'supercategory': 'food', 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
         {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
         {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
         {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
         {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
         {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
         {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
         {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
         {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
         {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
         {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
         {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
         {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
         {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
         {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
         {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
         {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
         {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
         {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
         {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
         {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
         {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'},
         {'supercategory': 'textile', 'id': 92, 'name': 'banner'},
         {'supercategory': 'textile', 'id': 93, 'name': 'blanket'}, {'supercategory': 'plant', 'id': 94, 'name': 'branch'},
         {'supercategory': 'building', 'id': 95, 'name': 'bridge'},
         {'supercategory': 'building', 'id': 96, 'name': 'building-other'},
         {'supercategory': 'plant', 'id': 97, 'name': 'bush'},
         {'supercategory': 'furniture-stuff', 'id': 98, 'name': 'cabinet'},
         {'supercategory': 'structural', 'id': 99, 'name': 'cage'},
         {'supercategory': 'raw-material', 'id': 100, 'name': 'cardboard'},
         {'supercategory': 'floor', 'id': 101, 'name': 'carpet'},
         {'supercategory': 'ceiling', 'id': 102, 'name': 'ceiling-other'},
         {'supercategory': 'ceiling', 'id': 103, 'name': 'ceiling-tile'},
         {'supercategory': 'textile', 'id': 104, 'name': 'cloth'},
         {'supercategory': 'textile', 'id': 105, 'name': 'clothes'}, {'supercategory': 'sky', 'id': 106, 'name': 'clouds'},
         {'supercategory': 'furniture-stuff', 'id': 107, 'name': 'counter'},
         {'supercategory': 'furniture-stuff', 'id': 108, 'name': 'cupboard'},
         {'supercategory': 'textile', 'id': 109, 'name': 'curtain'},
         {'supercategory': 'furniture-stuff', 'id': 110, 'name': 'desk-stuff'},
         {'supercategory': 'ground', 'id': 111, 'name': 'dirt'},
         {'supercategory': 'furniture-stuff', 'id': 112, 'name': 'door-stuff'},
         {'supercategory': 'structural', 'id': 113, 'name': 'fence'},
         {'supercategory': 'floor', 'id': 114, 'name': 'floor-marble'},
         {'supercategory': 'floor', 'id': 115, 'name': 'floor-other'},
         {'supercategory': 'floor', 'id': 116, 'name': 'floor-stone'},
         {'supercategory': 'floor', 'id': 117, 'name': 'floor-tile'},
         {'supercategory': 'floor', 'id': 118, 'name': 'floor-wood'},
         {'supercategory': 'plant', 'id': 119, 'name': 'flower'}, {'supercategory': 'water', 'id': 120, 'name': 'fog'},
         {'supercategory': 'food-stuff', 'id': 121, 'name': 'food-other'},
         {'supercategory': 'food-stuff', 'id': 122, 'name': 'fruit'},
         {'supercategory': 'furniture-stuff', 'id': 123, 'name': 'furniture-other'},
         {'supercategory': 'plant', 'id': 124, 'name': 'grass'}, {'supercategory': 'ground', 'id': 125, 'name': 'gravel'},
         {'supercategory': 'ground', 'id': 126, 'name': 'ground-other'},
         {'supercategory': 'solid', 'id': 127, 'name': 'hill'}, {'supercategory': 'building', 'id': 128, 'name': 'house'},
         {'supercategory': 'plant', 'id': 129, 'name': 'leaves'},
         {'supercategory': 'furniture-stuff', 'id': 130, 'name': 'light'},
         {'supercategory': 'textile', 'id': 131, 'name': 'mat'},
         {'supercategory': 'raw-material', 'id': 132, 'name': 'metal'},
         {'supercategory': 'furniture-stuff', 'id': 133, 'name': 'mirror-stuff'},
         {'supercategory': 'plant', 'id': 134, 'name': 'moss'}, {'supercategory': 'solid', 'id': 135, 'name': 'mountain'},
         {'supercategory': 'ground', 'id': 136, 'name': 'mud'}, {'supercategory': 'textile', 'id': 137, 'name': 'napkin'},
         {'supercategory': 'structural', 'id': 138, 'name': 'net'},
         {'supercategory': 'raw-material', 'id': 139, 'name': 'paper'},
         {'supercategory': 'ground', 'id': 140, 'name': 'pavement'},
         {'supercategory': 'textile', 'id': 141, 'name': 'pillow'},
         {'supercategory': 'plant', 'id': 142, 'name': 'plant-other'},
         {'supercategory': 'raw-material', 'id': 143, 'name': 'plastic'},
         {'supercategory': 'ground', 'id': 144, 'name': 'platform'},
         {'supercategory': 'ground', 'id': 145, 'name': 'playingfield'},
         {'supercategory': 'structural', 'id': 146, 'name': 'railing'},
         {'supercategory': 'ground', 'id': 147, 'name': 'railroad'}, {'supercategory': 'water', 'id': 148, 'name': 'river'},
         {'supercategory': 'ground', 'id': 149, 'name': 'road'}, {'supercategory': 'solid', 'id': 150, 'name': 'rock'},
         {'supercategory': 'building', 'id': 151, 'name': 'roof'}, {'supercategory': 'textile', 'id': 152, 'name': 'rug'},
         {'supercategory': 'food-stuff', 'id': 153, 'name': 'salad'},
         {'supercategory': 'ground', 'id': 154, 'name': 'sand'}, {'supercategory': 'water', 'id': 155, 'name': 'sea'},
         {'supercategory': 'furniture-stuff', 'id': 156, 'name': 'shelf'},
         {'supercategory': 'sky', 'id': 157, 'name': 'sky-other'},
         {'supercategory': 'building', 'id': 158, 'name': 'skyscraper'},
         {'supercategory': 'ground', 'id': 159, 'name': 'snow'},
         {'supercategory': 'solid', 'id': 160, 'name': 'solid-other'},
         {'supercategory': 'furniture-stuff', 'id': 161, 'name': 'stairs'},
         {'supercategory': 'solid', 'id': 162, 'name': 'stone'}, {'supercategory': 'plant', 'id': 163, 'name': 'straw'},
         {'supercategory': 'structural', 'id': 164, 'name': 'structural-other'},
         {'supercategory': 'furniture-stuff', 'id': 165, 'name': 'table'},
         {'supercategory': 'building', 'id': 166, 'name': 'tent'},
         {'supercategory': 'textile', 'id': 167, 'name': 'textile-other'},
         {'supercategory': 'textile', 'id': 168, 'name': 'towel'}, {'supercategory': 'plant', 'id': 169, 'name': 'tree'},
         {'supercategory': 'food-stuff', 'id': 170, 'name': 'vegetable'},
         {'supercategory': 'wall', 'id': 171, 'name': 'wall-brick'},
         {'supercategory': 'wall', 'id': 172, 'name': 'wall-concrete'},
         {'supercategory': 'wall', 'id': 173, 'name': 'wall-other'},
         {'supercategory': 'wall', 'id': 174, 'name': 'wall-panel'},
         {'supercategory': 'wall', 'id': 175, 'name': 'wall-stone'},
         {'supercategory': 'wall', 'id': 176, 'name': 'wall-tile'},
         {'supercategory': 'wall', 'id': 177, 'name': 'wall-wood'},
         {'supercategory': 'water', 'id': 178, 'name': 'water-other'},
         {'supercategory': 'water', 'id': 179, 'name': 'waterdrops'},
         {'supercategory': 'window', 'id': 180, 'name': 'window-blind'},
         {'supercategory': 'window', 'id': 181, 'name': 'window-other'},
         {'supercategory': 'solid', 'id': 182, 'name': 'wood'}]
    elif args.data_type == "voc":
        cats = {1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair",
                10: "cow", 11: "table", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
                16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}
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
    elif args.data_type == "ade20k":
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


    for file_dir in ['all_drop_hist_with_filtered_caption']: #'hist_withfiltered_caption',
        import os

        path = f"{args.save_path}/{file_dir}/"

        filename_list = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            filename_list.extend(filenames)
            break
        print("folder file len", len(filename_list))

        for i, filename in enumerate(filename_list):
            if i == 0:
                hist = np.load(os.path.join(path,filename),allow_pickle=True) #.item()
            else:
                hist += np.load(os.path.join(path,filename),allow_pickle=True) #.item()

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        valid = hist.sum(axis=1) > 0  # added
        mean_iu = np.nanmean(iu[valid])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        class_name_list = []


        if "coco" in args.data_type:
            if args.data_type == "coco_object":
                for class_id in range(91):
                    class_name_list.append(f"{class_id}_{getClassName(class_id, cats)}")
            elif args.data_type == "coco_stuff":
                for class_id in range(183):
                    class_name_list.append(f"{class_id}_{getClassName(class_id, cats)}")
        else:
            for class_id in range(len(cats)+1):
                if class_id == 0:
                    class_name_list.append("Background")
                else:
                    class_name_list.append(cats[int(class_id)])

        # print("69, iu.shape", iu.shape, class_name_list)
        cls_iu = dict(zip(class_name_list, iu))
        # cls_iu = dict(zip(range(n_class), iu))

        print(file_dir, {
            "Pixel Accuracy": acc,
            "Mean Accuracy": acc_cls,
            "Frequency Weighted IoU": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        })





if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main()