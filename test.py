import torch
import torch.nn as nn

from utils import top_accuracy
from dataset import CLRDataset
from model import ResNetSimCLR, ResNet

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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

TEST_LOADER = torch.utils.data.DataLoader(
    CLRDataset.get_dataset("cifar100-test", n_views=1, device=DEVICE),
    batch_size=512,
    shuffle=True,
)
TRAIN_LOADER = torch.utils.data.DataLoader(
    CLRDataset.get_dataset("cifar100-train", n_views=1, device=DEVICE),
    batch_size=512,
    shuffle=True,
)


def freeze_model(model):
    model.requires_grad_(False)
    model.backbone.fc.requires_grad_(True)


def redefine_fc_layer(model, target_dim=100):
    model.backbone.fc = nn.Linear(512, target_dim)  # 512 for resnet18


def test_model(model):
    top1_accuracy = 0
    top5_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(TEST_LOADER):
        x_batch = x_batch.to(DEVICE, dtype=torch.float)
        y_batch = y_batch.to(DEVICE)

        logits = model(x_batch)

        top1, top5 = top_accuracy(logits, y_batch, topk=(1, 5))
        top1_accuracy += top1[0]
        top5_accuracy += top5[0]

    top1_accuracy /= counter + 1
    top5_accuracy /= counter + 1
    return top1_accuracy, top5_accuracy


def train_model(epochs, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(TRAIN_LOADER):
            x_batch = x_batch.to(DEVICE, dtype=torch.float)
            y_batch = y_batch.to(DEVICE)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = top_accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= counter + 1
        top1_accuracy, top5_accuracy = test_model(model)
        print(
            f"Epoch {epoch}\tTop1 train_acc {top1_train_accuracy.item()}\tTop1 test_acc: {top1_accuracy.item()}\tTop5 test_acc: {top5_accuracy.item()}"
        )


def main():
    for setup in PREATRAINED_MODELS:
        weight_path, model_class = PREATRAINED_MODELS[setup]
        with open(weight_path, "rb") as f:
            checkpoint = torch.load(f)

        outdim = 200 if "transfer" in setup else 100
        model = model_class(outdim)
        model.load_state_dict(checkpoint["state_dict"])

        if setup == "super":
            # no train process
            model.to(device=DEVICE)
            top1_accuracy, top5_accuracy = test_model(model)
            print(
                f"@@@@ {setup.upper()} @@@@\nTop1 test_acc: {top1_accuracy.item()}\tTop5 test_acc: {top5_accuracy.item()}"
            )
        else:
            print(f"@@@@ {setup.upper()} @@@@")
            redefine_fc_layer(model)
            freeze_model(model)
            model.to(device=DEVICE)
            train_model(150, model)


if __name__ == "__main__":
    with torch.cuda.device(4):
        main()
