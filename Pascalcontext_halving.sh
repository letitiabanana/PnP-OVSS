#for head in {0,1,2,3,4,5,6,7,8,10,11}
#
#do
#  echo "head" $head
CUDA_VISIBLE_DEVICES=1,2 python pnp_get_attention_textloc_weaklysupervised_search_PSC.py \
--save_path Cbatch_Eval_test_ddp_1111_768_flickrfinetune_zeroshot_halvingdrop_PSC \
--master_port 22990 --gen_multiplecap_withpnpvqa label --world_size 2 \
--del_patch_num sort_thresh005 \
--img_size 768 \
--batch_size 28 \
--max_att_block_num 8 --drop_iter 5 --prune_att_head 9 --sort_threshold 0.05

#done
#CUDA_VISIBLE_DEVICES=2,3 nohup > > sort_thresh005_320_max8att9_wobackground_flikrfinetune_zeroshot.out --ensemble_blocks saveall
