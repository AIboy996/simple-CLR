import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, out_dim):
        super().__init__()
        self.backbone = models.resnet18(weights=None, num_classes=out_dim)
        dim_features = self.backbone.fc.in_features  # 512 for ResNet-18

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_features, dim_features),
            nn.ReLU(),
            self.backbone.fc,
        )

    def forward(self, x):
        return self.backbone(x)


class ResNet(nn.Module):

    def __init__(self, out_dim) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=None, num_classes=out_dim)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    import torch

    model = ResNetSimCLR(10)
    x = torch.rand((8, 3, 32, 32))
    print(x.shape)
    y = model(x)
    print(y.shape)
