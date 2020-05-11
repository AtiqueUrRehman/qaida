import sys
import torch
from torch import nn

from models.QRN18 import QRN18
from dataset import QaidaDataset
from torch.utils.data import DataLoader

from utils.framework import test_loop
from utils.io import parse_config

if __name__ == "__main__":
    config_path = "../data/config/app.json"

    config = parse_config(config_path)

    target_classes = config.get("target_classes")
    batch_size = config.get("test_batch_size")
    fc_neurons = config.get("fc_neurons", [])

    device = "cuda"
    train_dir = config.get("train_dir")
    test_dir = config.get("test_dir")

    best_path = config.get("best_path")

    model = QRN18(pre_trained=True, backbone="QRN18_18569", num_classes=target_classes,
                  model_config=config.get("model_config"),
                  fc_neurons=fc_neurons)
    print("Model Created")
    print(model)

    model.load_state_dict(torch.load(best_path))
    model.double()
    model.to(device)

    train_dataset = QaidaDataset(train_dir, transform=None, max_classes=target_classes)
    # Train dataloader should shuffle images
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Test dataset do not need transformations
    test_dataset = QaidaDataset(test_dir, transform=None, max_classes=target_classes)
    # Test dataloader should not shuffle images
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss, train_acc = test_loop(train_dataloader, model, criterion, device)

    print("Train Loss: {:.6f}".format(train_loss))
    print("Train Acc : {:.6f}".format(train_acc))

    test_loss, test_acc = test_loop(test_dataloader, model, criterion, device)

    print("Test Loss: {:.6f}".format(test_loss))
    print("Test acc: {:.6f}".format(test_acc))
