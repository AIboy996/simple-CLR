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

简称|训练算法|预训练数据集|评测方式|评测数据集
--|--|--|--|--
super-transfer|有监督|ImageNet|Linear Classification Protocol, transfer learning|CIFAR-100
selfsuper-transfer|自监督|ImageNet|Linear Classification Protocol, transfer learning|CIFAR-100
selfsuper|自监督|CIFAR-100|Linear Classification Protocol|CIFAR-100
super|有监督|CIFAR-100|直接评测|CIFAR-100

运行[data/get_dataset.sh](data/get_dataset.sh)自动下载、解压缩数据集。

运行[run_train.sh](run_train.sh)可以训练上述的四个模型。

运行[test.py](test.py)可以评测训练好的四个模型：
```python
PREATRAINED_MODELS = {
    "super-transfer": [
        f"runs/Jun25_15-00-{13}_user-AS-4124GS-TNR/checkpoint_0099.pth.tar",
        ResNet,
    ],
    "selfsuper-transfer": [
        f"runs/Jun25_15-00-{14}_user-AS-4124GS-TNR/checkpoint_0099.pth.tar",
        ResNetSimCLR,
    ],
    "selfsuper": [
        f"runs/Jun25_15-00-{16}_user-AS-4124GS-TNR/checkpoint_0099.pth.tar",
        ResNetSimCLR,
    ],
    "super": [
        f"runs/Jun25_15-00-{15}_user-AS-4124GS-TNR/checkpoint_0099.pth.tar",
        ResNet,
    ],
}
```

评测结果如下：
简称|评测方式|TOP1 accuracy| TOP5 accuracy
--|--|--|--
selfsuper-transfer|Linear Classification Protocol, transfer learning|32.93|61.08
super-transfer|Linear Classification Protocol, transfer learning|38.15|66.66
selfsuper|Linear Classification Protocol|35.77|63.55
super|直接评测|50.80|76.95

> TOP1 accuracy意为测试集中模型输出的概率中**最大的即为正确标签**的比例。
> 
> TOP5 accuracy意为测试集中模型输出的概率中**最大的前五个包含正确标签**的比例。
