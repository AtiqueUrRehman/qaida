import os
import PIL.Image as Image
import torchvision
from torch.utils.data import Dataset


class QaidaDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_classes=0):

        if transform is None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = transform

        self.image_paths, self.labels = self.init(data_dir, max_classes)
        self.__len = len(self.labels)

    def init(self, data_dir, max_classes):
        """
        Initialize labels and image_paths based on name of dirs
        """

        list_cls_dirs = sorted(os.listdir(data_dir), key=int)

        if max_classes == 0:
            print("Selecting max classes to : {}".format(len(list_cls_dirs)))

        elif 0 < max_classes < (len(list_cls_dirs) + 1):
            list_cls_dirs = list_cls_dirs[:max_classes]
            print("Selecting max classes to : {}".format(max_classes))

        else:
            print("Invalid arg max_classes: {}, Selecting max classes to : {}".format(max_classes, len(list_cls_dirs)))

        labels = []
        image_paths = []
        for cls_dir in list_cls_dirs:
            imgs_in_cls = [os.path.join(data_dir, cls_dir, img) for img in
                           os.listdir(os.path.join(data_dir, cls_dir))]

            num_imgs_in_cls = len(imgs_in_cls)
            cls_id = int(cls_dir)

            labels.extend([cls_id] * num_imgs_in_cls)
            image_paths.extend(imgs_in_cls)

        assert len(image_paths) == len(labels), "Size of images do not match size of labels"

        return image_paths, labels

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):

        lbl = int(self.labels[idx])

        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img.double(), lbl
