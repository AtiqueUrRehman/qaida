import torchvision
from torch import nn


class QRN18(nn.Module):
    def __init__(self, target_classes, pretrained = True):
        super(QRN18, self).__init__()
        self._model = torchvision.models.resnet18(pretrained=pretrained)

        if pretrained:
            # Freeze all feature extraction layers
            for param in self._model.parameters():
                param.requires_grad = False

        # Replace the prediction head
        fc = nn.Linear(512, target_classes)
        fc.requires_grad = True

        self._model.fc = fc
        self._model.fc.requires_grad = True

    def forward(self, images):
        return self._model(images)
