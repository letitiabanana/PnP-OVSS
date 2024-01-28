python New_eval_cam_pascalcontext.py \
--cam_out_dir ./Cbatch_Eval_test_ddp_1111_768_flickrfinetune_zeroshot_halvingdrop_PSC \
--save_path ./Pascalcontext_IOU/Pascalcontext_IOU_layer8head9_dropiter5_768_flickrfinetune_zeroshot_attthresh15_noblurnocrf \
--cam_threshold 0.15 --drop_patch_eval halving \
--img_size 768 \
--del_patch_num sort_thresh005 --drop_iter 5 \
--cam_att gradcam \
--max_att_block_num 8 --prune_att_head 9 \
--postprocess blur+crf \
--eval_mask_path ./cam_flood_fill_sort_thresh005_drop1_clipsim005_COCOfinetune_zeroshot_89_halving
# nohup > OUT/768_thresh005_drop3_clipsim25_edgemap &

