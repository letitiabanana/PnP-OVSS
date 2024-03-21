# Code Implementation for CVPR2024 PAPER -- Emergent Open-Vocabulary Semantic Segmentation from Off-the-shelf Vision-Language Models (PnP-OVSS)

:exclamation: Only the code for BLIP with Pascal Context and Cityscapes is provided here. 

## System 
CUDA Version: 11.7  <br>
GPU Memory: 49140MiB <br>


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
Pascal VOC <br>
Pascal Context: Download dataset following instruction from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context) <br>
COCO Object <br>
COCO Stuff <br>
ADE20K <br>
Cityscapes: Download dataset following instruction from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context) <br>


```
LAVIS
├── mmsegmentation
│   ├── VOCdevkit
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
├── Cityscapes
│   ├── leftImg8bit
│   │   ├── train
│   │   ├── val
│   ├── gtFine
│   │   ├── train
│   │   ├── val

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

For Pascal Context <br>
`bash PSC_halving.sh`

For Cityscapes <br>
`bash Cityscapes_halving.sh`


### For Gaussian Blur, Dense CRF and Mask Evalutaion

For COCO Object and COCO stuff <br>
`bash New_eval_cam_PSC.sh`

For Cityscapes <br>
`bash New_eval_cam_Cityscapes.sh`

The output would have the following structure
```
LAVIS
├── New_Cbatch_Eval_test_ddp_0126_768_flickrfinetune_zeroshot_halvingdrop_Cityscapes
│   ├── gradcam
│   │   ├── max_att_block_num8_del_patch_numsort_thresh005
│   │   │   ├── drop_iter0
│   │   │   │   ├──img_att_forclasses (attention map)
│   │   │   │   ├──Union_check0928    (visualization of attention map)
│   │   │   │   ├──highest_att_save   (index of patches to be dropped)
│   │   │   ├── drop_iter1
│   │   │   ├── drop_iter2
│   │   │   ├── drop_iter3
│   │   │   ├── drop_iter4
```


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
Remember to match the save_path in {xxx}\_halving.sh with the cam_out_dir in New_eval_cam\_{xx}.sh
