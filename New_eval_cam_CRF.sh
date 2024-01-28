python New_eval_cam_CRF.py \
    --cam_out_dir ./Cbatch_Eval_test_ddp_1021_combinelabelascaption_softmaxloss_cancelmax_byatthead_normalized_768_wobackground_flickrfinetune_zeroshot_cocoall_halvingdrop \
    --save_path ./COCO_CRF_refined_IOU/COCOall_Layer8_head9_dropiter5_768_flickrfinetune_zeroshot_cocostuff_attthresh15 \
    --data_type cocostuff \
    --img_size 768 \
    --cam_threshold 0.15 --drop_patch_eval halving \
    --del_patch_num sort_thresh005 --drop_iter 5 \
    --cam_att gradcam \
    --postprocess blur+crf \
    --max_att_block_num 8  --prune_att_head 9 \
    --eval_mask_path ./cam_flood_fill_sort_thresh005_drop1_clipsim005_COCOfinetune_zeroshot_89_halving
    # nohup > OUT/768_thresh005_drop3_clipsim25_edgemap &

