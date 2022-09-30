# lpn vgg16
#python train_cvusa.py \
#--name='usa_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
#--data_dir='/home/dm/datasets/CVUSA/train' \
#--warm_epoch=5 \
#--batchsize=8 \
#--h=256 \
#--w=256 \
#--use_vgg16 \
#--LPN \
#--warm_epoch=5 \
#--lr=0.05 \
#--block=8 \
#--gpu_ids='0'




# VGG16的效果
# my:     Recall@1:81.15 Recall@5:92.93 Recall@10:95.43 Recall@top1:98.82 AP:83.80 (epoch_100)
# my:     Recall@1:81.97 Recall@5:93.33 Recall@10:95.44 Recall@top1:98.98 AP:84.55 (epoch_120)
# my:     Recall@1:80.38 Recall@5:92.78 Recall@10:95.07 Recall@top1:98.85 AP:83.15 (epoch_140)
# my:     Recall@1:79.40 Recall@5:91.76 Recall@10:94.65 Recall@top1:98.64 AP:82.17 (epoch_160)
# my:     Recall@1:84.89 Recall@5:95.18 Recall@10:97.04 Recall@top1:99.39 AP:87.18 (epoch_180)
# paper:  Recall@1:79.69 Recall@5:91.70 Recall@10:94.55 Recall@top1:98.50

#python test_cvusa.py \
#--name='usa_vgg_noshare_warm5_8LPN-s-r_lr0.1' \
#--test_dir='/home/dm/datasets/CVUSA/val' \
#--batchsize=32 \
#--gpu_ids='0' \

# lpn resnet50
# python train_cvusa.py \
# --name='usa_res50_noshare_warm5_8LPN-s-r_lr0.05' \
# --data_dir='/home/wangtyu/datasets/CVUSA/train' \
# --warm_epoch=5 \
# --batchsize=16 \
# --h=256 \
# --w=256 \
# --fp16 \
# --LPN \
# --lr=0.05 \
# --block=8 \
# --stride=1 \
# --gpu_ids='0'

# resnet50的效果
# my:     Recall@1:85.40 Recall@5:95.08 Recall@10:96.80 Recall@top1:99.45 AP:87.55 (epoch_100)
# my:     Recall@1:85.74 Recall@5:95.29 Recall@10:97.16 Recall@top1:99.44 AP:87.88 (epoch_140)
# my:     Recall@1:85.84 Recall@5:94.96 Recall@10:96.87 Recall@top1:99.47 AP:87.92 (epoch_160)
# my:     Recall@1:87.16 Recall@5:95.98 Recall@10:97.55 Recall@top1:99.49 AP:89.15 (epoch_180)
# paper:  Recall@1:85.79 Recall@5:95.38 Recall@10:96.98 Recall@top1:99.41

python test_cvusa.py \
--name='usa_res50_noshare_warm5_8LPN-s-r_lr0.05' \
--test_dir='/home/dm/datasets/CVUSA/val' \
--batchsize=128 \
--gpu_ids='0'