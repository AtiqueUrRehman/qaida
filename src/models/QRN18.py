from collections import OrderedDict

import torch
import torchvision
from torch import nn


# TODO generic code for loading all layers except last fc
class QRN18(nn.Module):
    def __init__(self, num_classes, model_config, backbone="resnet18", pre_trained=True, freeze_backbone=False,
                 fc_neurons=[]):
        super(QRN18, self).__init__()
        if backbone == "resnet18":
            self._model = torchvision.models.resnet18(pretrained=pre_trained)

        elif backbone == "QRN18_400":
            self._model = torchvision.models.resnet18(pretrained=False)
            fc = nn.Linear(512, 400)
            self._model.fc = fc
            if pre_trained:
                self.load_state_dict(torch.load(model_config["QRN18_400"]))

        elif backbone == "QRN18_2000":
            self._model = torchvision.models.resnet18(pretrained=False)
            fc = nn.Linear(512, 2000)
            self._model.fc = fc
            if pre_trained:
                self.load_state_dict(torch.load(model_config["QRN18_2000"]))

        elif backbone == "QRN18_18569":
            self._model = torchvision.models.resnet18(pretrained=False)
            fc = nn.Linear(512, 18569)
            self._model.fc = fc
            if pre_trained:
                self.load_state_dict(torch.load(model_config["QRN18_18569"]))

        if freeze_backbone:
            # Freeze all feature extraction layers
            for param in self._model.parameters():
                param.requires_grad = False

        classifier_layers = []
        num_inputs = 512
        layer_id = 0
        for num_outputs in fc_neurons:
            classifier_layers.append((
                'fc{}'.format(layer_id if layer_id else ""),
                nn.Linear(num_inputs, num_outputs)
            ))
            num_inputs = num_outputs
            layer_id += 1

        classifier_layers.append((
            'fc{}'.format(layer_id if layer_id else ""),
            nn.Linear(num_inputs, num_classes)
        ))

        classifier = nn.Sequential(OrderedDict(classifier_layers))

        self._model.fc = classifier
        self._model.fc.requires_grad = True

    def forward(self, images):
        return self._model(images)
