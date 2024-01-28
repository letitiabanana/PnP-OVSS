#for head in {0,1,2,3,4,5,6,7,8,10,11}
#
#do
#  echo "head" $head
CUDA_VISIBLE_DEVICES=3 python pnp_get_attention_textloc_weaklysupervised_search_Cityscapes.py \
--save_path New_Cbatch_Eval_test_ddp_0126_768_flickrfinetune_zeroshot_halvingdrop_Cityscapes \
--master_port 10990 --gen_multiplecap_withpnpvqa label --world_size 1 \
--del_patch_num sort_thresh005 \
--img_size 448 \
--batch_size 2 \
--max_att_block_num 8 --drop_iter 5 --prune_att_head 9 --sort_threshold 0.05

#done
#CUDA_VISIBLE_DEVICES=2,3 nohup > > sort_thresh005_768_max8att9_wobackground_flikrfinetune_zeroshot.out --ensemble_blocks saveall
