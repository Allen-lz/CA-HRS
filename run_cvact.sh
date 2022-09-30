# # LPN vgg16
# python train_cvact.py \
# --name='act_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
# --data_dir='/home/wangtyu/datasets/CVACT/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --use_vgg16 \
# --h=256 \
# --w=256 \
# --LPN \
# --lr=0.1 \
# --block=8 \
# --gpu_ids='2'

python test_cvact.py \
 --name='act_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
 --test_dir='/home/dm/datasets/CVACT/val' \
 --gpu_ids='0'


# Recall@1:80.91 Recall@5:90.95 Recall@10:92.93 Recall@top1:97.07 AP:83.20

# LPN resnet50
#python train_cvact.py \
#--name='act_res50_noshare_warm5_8LPN-s-r_lr0.05' \
#--data_dir='/home/dm/datasets/CVACT/train' \
#--warm_epoch=5 \
#--batchsize=16 \
#--h=256 \
#--w=256 \
#--LPN \
#--lr=0.05 \
#--block=8 \
#--stride=1 \
#--gpu_ids='0'

#python test_cvact.py \
#--name='act_res50_noshare_warm5_8LPN-s-r_lr0.05' \
#--test_dir='/home/dm/datasets/CVACT/val' \
#--batchsize=64 \
#--gpu_ids='0'