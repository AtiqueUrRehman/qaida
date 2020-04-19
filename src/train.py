import sys
import numpy as np

import torch
import torchvision
from torch import optim, nn
from tqdm import tqdm

from model import QRN18
from dataset import QaidaDataset
from torch.utils.data import DataLoader
from transform import get_transform


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    training_epochs = 20
    target_classes = 2000
    train_batch_size = 512
    test_batch_size = 512
    start_lr = 0.002

    device = "cuda"
    train_dir = "../data/train_20k"
    test_dir = "../data/test_20k"
    save_path = "../data/models/200_iter{}.bin"
    best_path = "../data/models/200_best.bin"

    model = QRN18(target_classes=target_classes)
    model.double()
    model.to(device)

    # Lets add some transformation to make our model translation and scale invarient
    to_tensor = torchvision.transforms.ToTensor()

    train_dataset = QaidaDataset(train_dir, transform=get_transform(mode="train"), max_classes = 400)
    # Train dataloader should shuffle images
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    # Test dataset do not need transformations
    test_dataset = QaidaDataset(test_dir, transform=get_transform(mode="test"), max_classes = 400)
    # Test dataloader should not shuffle images
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_parameters, lr=start_lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    sys.stdout.flush()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = 200
    train_loss = []
    min_loss = np.inf

    for e in range(epochs):
        # Train loss
        model.train()
        running_loss = 0
        for imgs, lbls in tqdm(train_dataloader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            pred = model(imgs)

            loss = criterion(pred, lbls)
            running_loss += loss
            loss.backward()
            optimizer.step()
        running_loss /= len(train_dataloader)

        print("Epoch : {}/{}..".format(e + 1, epochs),
              "Training Loss: {:.6f}".format(running_loss))
        train_loss.append(running_loss)

        lr_scheduler.step(running_loss)

        # Test loss
        model.eval()
        test_loss = 0
        for imgs, lbls in tqdm(test_dataloader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            pred = model(imgs)

            loss = criterion(pred, lbls)
            test_loss += loss
        test_loss /= len(test_dataloader)

        print("Epoch : {}/{}..".format(e + 1, epochs),
              "Test Loss: {:.6f}".format(test_loss))
        print("Learning after epoch {} is rate {}".format(e + 1, get_lr(optimizer)))

        torch.save(model.state_dict(), save_path.format(e))
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), best_path)
            print("Best model saved with test loss {}".format(min_loss))

        sys.stdout.flush()
