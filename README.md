# Open-Vocab-Semantic-Segmentation

[!]Only the code of PnP_OVSS + BLIP for Pascal Context and Cityscapes is provided here. I haven't clean up my code yet so the structure is still messy sryyyy~~

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
This would download the newest torch, may need to modify torch version based on your cuda version.
Might also need to downgrade transformer
`pip install transformers==4.25`

## Download Gradient_Free_Optimizers_master

Download [Gradient_Free_Optimizers_master](https://github.com/SimonBlanke/Gradient-Free-Optimizers) and put it under LAVIS (This is for Random Search. Can ignore it for now)


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
Download all the files in this repository and put them under LAVIS
Replace /home/user/LAVIS/lavis/models/blip_models/blip_image_text_matching.py with the file in this repository <br>
Replace /home/user/LAVIS/lavis/configs/models/blip_itm_large.yaml with the file in this repository <br>
Replace /home/user/LAVIS/lavis/models/blip_models/med.py with the file in this repository <br>
Replace /home/user/LAVIS/lavis/models/blip_models/vit.py with the file in this repository <br>
Replace /home/user/LAVIS/lavis/models/blip_models/base_model.py with the file in this repository <br>
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
