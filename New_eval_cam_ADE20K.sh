python New_eval_cam_ADE20K.py \
--cam_out_dir ./Test_ADE20K \
--save_path ./Test_COCO_CRF_refined_IOU/ADE20K_IOU_layer8head9_dropiter5_768_cocofinetune_zeroshot_attthresh15_blurcrf \
--cam_threshold 0.1 --drop_patch_eval halving \
--img_size 768 \
--del_patch_num sort_thresh005 --drop_iter 5 \
--cam_att gradcam \
--max_att_block_num 8 --prune_att_head 9 \
--postprocess blur+crf \
--eval_mask_path ./cam_flood_fill_sort_thresh005_drop1_clipsim005_COCOfinetune_zeroshot_89_halving

# nohup > OUT/768_thresh005_drop3_clipsim25_M.BSedgemap &
