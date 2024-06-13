# Code Implementation for CVPR2024 PAPER -- Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models (PnP-OVSS)

:exclamation: This is the updated verision of PnP-OVSS with the bug corrected. Please refer to our latest arxiv version of the paper for the updated results and procedures. Only the code for PnP-OVSS + BLIP is provided here. Will update BridgeTower shortly.

## System 
CUDA Version: 11.7  <br>
GPU Memory: 48GB <br>


## Download LAVIS
Build LAVIS environment following the instruction [here](https://www.google.com](https://github.com/salesforce/LAVIS/tree/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc?tab=readme-ov-file#installation)https://github.com/salesforce/LAVIS/tree/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc?tab=readme-ov-file#installation)
```
conda create -n lavis python=3.8 
conda activate lavis 
pip install salesforce-lavis 
git clone https://github.com/salesforce/LAVIS.git 
cd LAVIS 
pip install -e .
```
This would download the newest torch, may need to modify torch version based on your cuda version. <br>
Might also need to downgrade transformer
`pip install transformers==4.25`

## Download Gradient_Free_Optimizers_master

Download [Gradient_Free_Optimizers_master](https://github.com/SimonBlanke/Gradient-Free-Optimizers) and put it under LAVIS (This is for Random Search. Can ignore it for now)

## Download pydensecrf

Git clone [pydensecrf](https://github.com/lucasb-eyer/pydensecrf) and put it under LAVIS 

## Download datasets

ADE20K Download dataset from [here](http://groups.csail.mit.edu/vision/datasets/ADE20K/request_data/) <br>
May Download the following datasets following instruction from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context) <br>
Pascal VOC <br>
Pascal Context <br>
COCO Object <br>
COCO Stuff <br>



```
LAVIS
├──mmsegmentation
│   ├── mmseg
│   ├── tools
│   ├── configs
│   ├── data
│   │   ├── VOCdevkit
│   │   │   ├── VOC2012
│   │   │   │   ├── JPEGImages
│   │   │   │   ├── SegmentationClass
│   │   │   │   ├── Annotations
│   │   │   ├── VOC2010
│   │   │   │   ├── JPEGImages
│   │   │   │   ├── SegmentationClassContext
│   │   │   │   ├── ImageSets
│   │   │   │   │   ├── SegmentationContext
│   │   │   │   │   │   ├── train.txt
│   │   │   │   │   │   ├── val.txt
│   │   │   │   ├── trainval_merged.json
│   │   ├── coco_stuff164k
│   │   │   ├── images
│   │   │   │   ├── train2017
│   │   │   │   ├── val2017
│   │   │   ├── annotations
│   │   │   │   ├── train2017
│   │   │   │   ├── val2017
├── coco
│   ├── images
│   │   ├── val2017
├── ADEChallengeData2016
│   ├── images
│   │   ├── validation
│   ├── annotations
│   │   ├── validation


```

## Replace the config file and GradCAM file of BLIP
Download all the files in this repository and put them under LAVIS <br>
Replace /home/user/LAVIS/lavis/models/blip_models/blip_image_text_matching.py with the file in this repository <br>
Replace /home/user/LAVIS/lavis/configs/models/blip_itm_large.yaml with the file in this repository <br>
Replace /home/user/LAVIS/lavis/models/med.py with the file in this repository <br>
Replace /home/user/LAVIS/lavis/models/vit.py with the file in this repository <br>
Replace /home/user/LAVIS/lavis/models/base_model.py with the file in this repository <br>
Replace /home/user/LAVIS/lavis/processors/blip_processors.py with the file in this repository <br>


## Obtain the GPT4o Classification
We provide the classification result from GPT4o for the five datasets in this repository, if you would like to run GPT4o by yourself, you may obtain your GPT KPI Key and run:

```
python PnP_OVSS_0514_updated_segmentation.py \
  --apikey xxxx
  --output_dir GPT4o_classification
  --data_type coco_object 
```


## Run scripts

### For Pascal VOC, Pascal Context, and ADE20K, modify the data_type to "voc", "psc", and "ade20k" <br>
You may modify world_size to run on multiple GPUs <br>

Run `bash Run_seg.sh`
```
python PnP_OVSS_0514_updated_segmentation.py \
  --home_dir \home\letitiabanana\LAVIS \
  --save_path BLIP_0602_336_ADE20K_segmentation\
  --master_port 29790 --gen_multiplecap_withpnpvqa label --world_size 1 \
  --img_size 336 \
  --del_patch_num sort_thresh005 \
  --batch_size 35 \
  --max_att_block_num 8 --drop_iter 4 --prune_att_head 9 --sort_threshold 0.05 \
  --threshold 0.15 \
  --postprocess blur+crf \
  --data_type ade20k
```
Then modify the save_path and data_type in "Calculate_mIoU.sh" and run `bash calculate_mIou.sh` to get the final result for the datasets

```
python Calculate_mIoU.py \
--save_path BLIP_0602_336_ADE20K_segmentation \
--data_type ade20k
```


### For COCO Object and COCO Stuff, set the data_typa argument in the bash file as "coco_object" or "coco_stuff"  <br>

run `bash Run_seg_coco.sh`
```
python PnP_OVSS_0514_updated_segmentation_coco.py \
  --home_dir \home\letitiabanana\LAVIS \
  --save_path BLIP_0602_336_ADE20K_segmentation\
  --master_port 29790 --gen_multiplecap_withpnpvqa label --world_size 1 \
  --img_size 336 \
  --del_patch_num sort_thresh005 \
  --batch_size 35 \
  --max_att_block_num 8 --drop_iter 4 --prune_att_head 9 --sort_threshold 0.05 \
  --threshold 0.15 \
  --postprocess blur+crf \
  --data_type coco_object
```
Then modify the save_path and data_type in "Calculate_mIoU.sh" and run `bash calculate_mIou.sh` to get the final result for the datasets




For modifying image size, you may also need to modify the image size in /home/user/LAVIS/lavis/configs/models/blip_itm_large.yaml <br>


## Please cite us with this BibTeX:
```
@article{luo2023emergent,
  title={Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models},
  author={Luo, Jiayun and Khandelwal, Siddhesh and Sigal, Leonid and Li, Boyang},
  journal={arXiv e-prints},
  pages={arXiv--2311},
  year={2023}
}
```
