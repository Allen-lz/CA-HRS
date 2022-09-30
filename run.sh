# CA-HRS
#python train.py \
#--name='final_three_view_long_share_d0.75_256_s1_google_LPN4_lr0.001_wp5' \
#--data_dir='D:/datasets/University-Release/train' \
#--views=3 \
#--droprate=0.75 \
#--extra \
#--share \
#--stride=1 \
#--h=256 \
#--w=256 \
#--LPN \
#--block=4 \
#--warm_epoch=5 \
#--lr=0.001 \
#--gpu_ids='0'


# my: Recall@1:86.88 Recall@5:90.44 Recall@10:92.44 Recall@top1:98.72 AP:74.83  satellite->drone
# my: Recall@1:76.67 Recall@5:90.36 Recall@10:93.44 Recall@top1:93.76 AP:79.77  drone->satellite
python test.py \
--name='final_three_view_long_share_d0.75_256_s1_google_LPN4_lr0.001_wp5' \
--test_dir='/home/dm/datasets/University-Release/test' \
--batchsize=128 \
--gpu_ids='0'



# python test.py --name=final_three_view_long_share_d0.75_256_s1_google_LPN4_lr0.001_wp5 --test_dir=/data/luzeng/datasets/University_1652/test --batchsize=128 --gpu_ids=0