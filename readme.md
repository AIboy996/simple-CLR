# simple-clr

神经网络和深度学习（DATA620004）期末作业，任务1：对比**监督学习**和**自监督学习**在图像分类任务上的性能表现.

## 要求
- 实现任一自监督学习算法并使用该算法在自选的数据集上训练ResNet-18，随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测；
- 将上述结果与在ImageNet数据集上采用监督学习训练得到的表征在相同的协议下进行对比，并比较二者相对于在CIFAR-100数据集上从零开始以监督学习方式进行训练所带来的提升；
- 尝试不同的超参数组合，探索自监督预训练数据集规模对性能的影响；

## 实现

- 模型：[ResNet-18](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html)（输出的特征纬度：512x512）
- 使用的自监督学习算法：[SimCLR - A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/google-research/simclr)
- 预训练数据集：[Tiny ImageNet 200](https://www.image-net.org/)
    - **200 classes, 600(=500+50+50) 64×64 colored images for each class**
- 评测数据集：[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
    - **100 classes, 600(=500+100) 32x32 colored images for each class**

## 模型训练

训练算法|预训练数据集|评测方式|评测数据集
--|--|--|--
监督训练|ImageNet|Linear Classification Protocol|CIFAR-100
自监督训练|ImageNet|Linear Classification Protocol|CIFAR-100
监督训练|CIFAR-100|直接评测|CIFAR-100

运行[data/get_dataset.sh](data/get_dataset.sh)自动下载、解压缩数据集。