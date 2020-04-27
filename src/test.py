import sys
import numpy as np

import torch
import torchvision
from torch import optim, nn
from tqdm import tqdm

from model import QRN18
from dataset import QaidaDataset
from torch.utils.data import DataLoader

def calculate_accuracy(pred, lbls):
    acc = sum(np.argmax(pred.to('cpu'), axis = 1) == lbls.to('cpu')) / float(lbls.to('cpu').shape[0])
    return acc

@torch.no_grad()
def test_loop(dataloader, model, criterian, device):
    
    model.eval()
    
    test_loss = 0.0
    total_acc = 0.0

    for imgs, lbls in tqdm(dataloader):
        imgs, lbls = imgs.to(device), lbls.to(device)

        pred = model(imgs)
        acc = calculate_accuracy(pred, lbls)

        loss = criterion(pred, lbls)

        total_acc += acc
        test_loss += loss

    test_loss /= len(dataloader)
    total_acc /= len(dataloader)
    
    return test_loss, total_acc


if __name__ == "__main__":
    target_classes = 400
    train_batch_size = 128
    test_batch_size = 128

    device = "cuda"
    train_dir = "../../qaida/data/train_20k"
    test_dir = "../../qaida/data/test_20k"
    best_path = "../../qaida/data/400_scratch_best_2.bin"

    model = QRN18(pretrained = False, target_classes=target_classes)
    model.load_state_dict(torch.load(best_path))
    model.double()
    model.to(device)

    # Lets add some transformation to make our model translation and scale invarient
    to_tensor = torchvision.transforms.ToTensor()

    train_dataset = QaidaDataset(train_dir, transform=None, max_classes = target_classes)
    # Train dataloader should shuffle images
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    # Test dataset do not need transformations
    test_dataset = QaidaDataset(test_dir, transform=None, max_classes = target_classes)
    # Test dataloader should not shuffle images
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss = []
    min_loss = np.inf

    train_loss, train_acc = test_loop(train_dataloader, model, criterion, device)
    
    print("Train Loss: {:.6f}".format(train_loss))
    print("Train Acc : {:.6f}".format(train_acc))

    test_loss, test_acc = test_loop(test_dataloader, model, criterion, device)

    print("Test Loss: {:.6f}".format(test_loss))
    print("Test acc: {:.6f}".format(test_acc))
