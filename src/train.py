import sys
import numpy as np

import torch
import torchvision
from torch import optim, nn
from tqdm import tqdm

from utils.io import parse_config
from models.QRN18 import QRN18
from dataset import QaidaDataset
from torch.utils.data import DataLoader
from utils.transform import get_transform
from utils.framework import calculate_accuracy, test_loop, get_lr


def get_model(classes, fc, model_config):
    if classes == 400:
        return QRN18(pre_trained=True, backbone="resnet18", num_classes=target_classes, model_config=model_config,
                     fc_neurons=fc_neurons)
    elif classes == 2000:
        return QRN18(pre_trained=True, backbone="QRN18_400", num_classes=target_classes, model_config=model_config,
                     fc_neurons=fc_neurons)
    elif classes == 18569:
        return QRN18(pre_trained=True, backbone="QRN18_18569", num_classes=target_classes, model_config=model_config, 
                     fc_neurons=fc_neurons)


if __name__ == "__main__":
    config_path = "../data/config/app.json"

    config = parse_config(config_path)

    epochs = config.get("epochs", 200)
    target_classes = config.get("target_classes")
    train_batch_size = config.get("train_batch_size")
    test_batch_size = config.get("test_batch_size")
    start_lr = config.get("start_lr")
    weight_decay = config.get("weight_decay")
    restart_from_epoch = config.get("restart_from_epoch")
    fc_neurons = config.get("fc_neurons", [])

    device = "cuda"
    train_dir = config.get("train_dir")
    test_dir = config.get("test_dir")

    save_path = config.get("save_path")
    best_path = config.get("best_path")

    model = get_model(target_classes, fc_neurons, config.get("model_config"))
    print ("Model Created")
    print (model)

    if restart_from_epoch:
        model.load_state_dict(torch.load(save_path.format(restart_from_epoch - 1)))
        print("Loading state from epoch: {}".format(restart_from_epoch - 1))

    model.double()
    model.to(device)

    # Lets add some transformation to make our model translation and scale invarient
    to_tensor = torchvision.transforms.ToTensor()

    train_dataset = QaidaDataset(train_dir, transform=get_transform(mode="train"), max_classes=target_classes)
    # Train dataloader should shuffle images
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    # Test dataset do not need transformations
    test_dataset = QaidaDataset(test_dir, transform=get_transform(mode="test"), max_classes=target_classes)
    # Test dataloader should not shuffle images
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_parameters, lr=start_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    sys.stdout.flush()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    min_loss = np.inf

    for e in range(restart_from_epoch, epochs):
        # Train loop
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for imgs, lbls in tqdm(train_dataloader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            pred = model(imgs)

            loss = criterion(pred, lbls)
            acc = calculate_accuracy(pred, lbls)

            running_loss +=  float(loss)
            running_acc += float(acc)
            loss.backward()
            optimizer.step()

        running_loss /= len(train_dataloader)
        running_acc /= len(train_dataloader)

        print("Epoch : {}/{}..".format(e + 1, epochs),
              "Training Loss: {:.6f}".format(running_loss),
              "Training Acc ; {:.6f}".format(running_acc))

        lr_scheduler.step(running_loss)

        # Test loop
        model.eval()

        test_loss, test_acc = test_loop(test_dataloader, model, criterion, device)

        print("Epoch : {}/{}..".format(e + 1, epochs),
              "Test Loss: {:.6f}".format(test_loss),
              "Test Acc : {:.6f}".format(test_acc))

        print("Learning after epoch {} is rate {}".format(e + 1, get_lr(optimizer)))

        torch.save(model.state_dict(), save_path.format(e))
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), best_path)
            print("Best model saved after {} epoch with test loss {:.6f} and test acc {:.6f}".format(e + 1, min_loss,
                                                                                                     test_acc))

        sys.stdout.flush()
