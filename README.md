# Open-Vocab-Semantic-Segmentation

## Download LAVIS
Build LAVIS environment following the instruction [here]([https://www.google.com](https://github.com/salesforce/LAVIS/tree/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc?tab=readme-ov-file#installation)https://github.com/salesforce/LAVIS/tree/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc?tab=readme-ov-file#installation)

## Download datasets
Pascal VOC \\ 
Pascal Context: Download dataset following instruction from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context)
COCO Object \\
COCO Stuff \\
ADE20K \\
Cityscapes: Download dataset following instruction from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context)

## Replace the config file and GradCAM file of BLIP
Replace /home/user/LAVIS/lavis/models/blip_models/blip_image_text_matching.py with the file in this repository
Replace /home/user/LAVIS/lavis/configs/models/blip_itm_large.yaml with the file in this repository

## Run scripts
### For saving the GradCAM maps and index of patches to drop for each round of Salience Drop
For COCO Object and COCO stuff
''' 
bash pnp_get_attention_halving.sh
'''
For Cityscapes
''' 
bash Cityscapes_halving.sh
'''
### For Gaussian Blur, Dense CRF and evalutaion
For COCO Object and COCO stuff
''' 
New_eval_cam_CRF.sh
'''
For Cityscapes
''' 
bash New_eval_cam_Cityscapes.sh
'''

