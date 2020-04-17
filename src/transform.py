import torchvision


def get_transform(mode):
    if mode == "train:":
        to_tensor = torchvision.transforms.ToTensor
        affine_transform = torchvision.transforms.RandomAffine(degrees=(0, 0),
                                                               translate=(0.1, 0.2),
                                                               scale=(1.1, 1.2),
                                                               fillcolor=(255, 255, 255))
        return torchvision.transforms.Compose([affine_transform, to_tensor])
    elif mode in ["valid", 'test']:
        return None
    else:
        NotImplementedError(" Only train, test and valid modes are implemented")
