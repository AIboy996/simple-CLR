epoch=100
# 在imagenet200上 有监督训练 200个epoch
python train.py -a ResNet -dataset-name imagenet200-train --n-views 1 --trainer SupervisedTrainer --gpu-index 1 --out_dim 200 --epoch $epoch &
sleep 1
# 在imagenet200上 自监督训练 200个epoch
python train.py -a ResNetSimCLR -dataset-name imagenet200-train --n-views 2 --trainer SelfSupervisedTrainer --gpu-index 2 --out_dim 200 --epoch $epoch &
sleep 1
# 在cirfar100上   有监督训练 200个epoch
python train.py -a ResNet -dataset-name cifar100-train --n-views 1 --trainer SupervisedTrainer --gpu-index 3 --out_dim 100 --epoch $epoch &
sleep 1
# 在cirfar100上   自监督训练 200个epoch
python train.py -a ResNetSimCLR -dataset-name cifar100-train --n-views 2 --trainer SelfSupervisedTrainer --gpu-index 4 --out_dim 100 --epoch $epoch &
