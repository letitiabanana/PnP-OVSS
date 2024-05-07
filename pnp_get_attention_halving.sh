#for head in {0,1,2,3,4,5,6,7,8,10,11}
#
#do
#  echo "head" $head
python pnp_get_attention_textloc_weaklysupervised_search.py \
  --save_path Test_cocothing \
  --master_port 19790 --gen_multiplecap_withpnpvqa label --world_size 1 \
  --img_size 768 \
  --del_patch_num sort_thresh005 \
  --batch_size 2 \
  --data_type cocothing \
  --max_att_block_num 8 --drop_iter 5 --prune_att_head 9 --sort_threshold 0.05
#done
#CUDA_VISIBLE_DEVICES=2,3 nohup > > sort_thresh005_336_max8att9_wobackground_flikrfinetune_zeroshot.out --ensemble_blocks saveall