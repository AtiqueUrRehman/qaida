import torch
import numpy as np
from tqdm import tqdm


def get_lr(optimizer):
    """
    Extract and return current learning rate from optimizer param_group
    :param optimizer: torch.optim
    :return: learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


@torch.no_grad()
def calculate_accuracy(pred, lbls):
    """
    Calculate accuracy from the predictions and ground truth labels
    :param pred: Tensor [n, c] where n is the number of samples and c is the number of classes
    :param lbls: Tensor [n] where n is the number of samples
    :return:
    """
    acc = float(sum(np.argmax(pred.to('cpu'), axis=1) == lbls.to('cpu')) / float(lbls.to('cpu').shape[0]))
    return acc


@torch.no_grad()
def test_loop(dataloader, model, criterion, device):
    """
    Loop over the dataset in eval mode with no gradients. Calculate criterion and accuracy
    :param dataloader: Torch Dataloader
    :param model: Torch.nn.Module
    :param criterion: Criterion from Torch.nn
    :param device: from ["cpu", "cuda"]
    :return: loss and criterion result
    """
    model.eval()

    test_loss = 0.0
    total_acc = 0.0

    for imgs, lbls in tqdm(dataloader):
        imgs, lbls = imgs.to(device), lbls.to(device)

        pred = model(imgs)
        acc = calculate_accuracy(pred, lbls)

        loss = criterion(pred, lbls)

        total_acc += float(acc)
        test_loss += float(loss)

    test_loss /= len(dataloader)
    total_acc /= len(dataloader)

    return test_loss, total_acc
