# Open-Vocab-Semantic-Segmentation

[!]Only the code of PnP_OVSS + BLIP for Pascal Context and Cityscapes is provided here. I haven't clean my code and combine the datasets yet so the code is still messy sryyyy~~

## Download LAVIS
Build LAVIS environment following the instruction [here](https://www.google.com](https://github.com/salesforce/LAVIS/tree/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc?tab=readme-ov-file#installation)https://github.com/salesforce/LAVIS/tree/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc?tab=readme-ov-file#installation)
`
conda create -n lavis python=3.8
conda activate lavis
pip install salesforce-lavis
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
`

## Download datasets
Pascal VOC <br>
Pascal Context: Download dataset following instruction from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context) <br>
COCO Object <br>
COCO Stuff <br>
ADE20K <br>
Cityscapes: Download dataset following instruction from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context) <br>

## Replace the config file and GradCAM file of BLIP
Replace /home/user/LAVIS/lavis/models/blip_models/blip_image_text_matching.py with the file in this repository <br>
Replace /home/user/LAVIS/lavis/configs/models/blip_itm_large.yaml with the file in this repository <br>

## Run scripts
### For saving the GradCAM maps and index of patches to drop for each round of Salience Drop

For COCO Object and COCO stuff <br>
`bash pnp_get_attention_halving.sh`

For Cityscapes <br>
` bash Cityscapes_halving.sh`


### For Gaussian Blur, Dense CRF and evalutaion

For COCO Object and COCO stuff <br>
`New_eval_cam_CRF.sh`

For Cityscapes <br>
`bash New_eval_cam_Cityscapes.sh`
