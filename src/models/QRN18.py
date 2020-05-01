import torch
import torchvision
from torch import nn


# TODO param for backbone type
class QRN18(nn.Module):
    def __init__(self, target_classes, backbone="resnet18", pre_trained=True, freeze_backbone=False):
        super(QRN18, self).__init__()
        if backbone == "resnet18":
            self._model = torchvision.models.resnet18(pretrained=pre_trained)

        elif backbone == "QRN18_400":
            self._model = torchvision.models.resnet18(pretrained=False)
            fc = nn.Linear(512, 400)
            self._model.fc = fc
            if pre_trained:
                self.load_state_dict(torch.load("../../qaida/data/400_scratch_best.bin"))

        elif backbone == "QRN18_2000":
            self._model = torchvision.models.resnet18(pretrained=False)
            fc = nn.Linear(512, 2000)
            self._model.fc = fc
            if pre_trained:
                self.load_state_dict(torch.load("../../qaida/data/2000_scratch_best.bin"))

        if freeze_backbone:
            # Freeze all feature extraction layers
            for param in self._model.parameters():
                param.requires_grad = False

        # Replace the prediction head
        fc = nn.Linear(512, target_classes)

        self._model.fc = fc
        self._model.fc.requires_grad = True

    def forward(self, images):
        return self._model(images)
