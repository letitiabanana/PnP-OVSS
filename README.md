# Code Implementation for CVPR2024 PAPER -- Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models (PnP-OVSS)

:exclamation: Only the code for PnP-OVSS + BLIP is provided here. Will update BridgeTower shortly.

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
May Download dataset following instruction from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context)
Pascal VOC <br>
Pascal Context <br>
COCO Object <br>
COCO Stuff <br>
ADE20K <br>


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

## Run scripts

### For saving the GradCAM maps and index of patches to drop for each round of Salience Drop
For Pascal VOC <br>
`bash VOC_halving.sh`

For Pascal Context <br>
`bash PSC_halving.sh`

For COCO Object and COCO Stuff, set the data_typa argument in the bash file as "cocothing" or "cocoall"  <br>
`bash pnp_get_attention_halving.sh`


For ADE20K  <br>
`bash ADE20K_halving.sh`



## Modify Hyperparameters in bash files
```
CUDA_VISIBLE_DEVICES=3 python pnp_get_attention_textloc_weaklysupervised_search_Cityscapes.py \
--save_path New_Cbatch_Eval_test_ddp_0126_448_flickrfinetune_zeroshot_halvingdrop_Cityscapes \
--master_port 10990 --gen_multiplecap_withpnpvqa label --world_size 1 \
--del_patch_num sort_thresh005 \
--img_size 768 \
--batch_size 2 \
--max_att_block_num 8 --drop_iter 5 --prune_att_head 9 --sort_threshold 0.05
```
To change image size, you may also need to modify the image size in /home/user/LAVIS/lavis/configs/models/blip_itm_large.yaml <br>


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
