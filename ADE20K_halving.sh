
CUDA_VISIBLE_DEVICES=0 python pnp_get_attention_textloc_weaklysupervised_search_ADE20K.py \
--save_path Test_ADE20K \
--master_port 22990 --gen_multiplecap_withpnpvqa label --world_size 1 \
--del_patch_num sort_thresh005 \
--img_size 768 \
--batch_size 2 \
--max_att_block_num 8 --drop_iter 5 --prune_att_head 9 --sort_threshold 0.05

#CUDA_VISIBLE_DEVICES=2,3 nohup > > sort_thresh005_320_max8att9_wobackground_flikrfinetune_zeroshot.out --ensemble_blocks saveall