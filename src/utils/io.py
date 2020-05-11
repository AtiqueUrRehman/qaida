import json
import os
from json import JSONDecodeError

import numpy as np


def get_images_from_dir(class_dir):
    """

    :param class_dir:
    :return:
    """
    return [os.path.join(class_dir, path) for path in sorted(os.listdir(class_dir))]


def get_data_numpy(data_dir, class_list, num_images_per_class=10, randomise=True, flatten=True):
    """

    :param data_dir:
    :param class_list:
    :param num_images_per_class:
    :param randomise:
    :param flatten:
    :return:
    """
    if flatten:
        dim = (80 * 80,)
    else:
        dim = (80, 80)

    np_data = np.ones((len(class_list) * num_images_per_class, *dim), dtype=np.float64)
    labels = np.ones(len(class_list) * num_images_per_class, dtype=np.int64)

    idx = 0
    for c in class_list:
        image_paths = get_images_from_dir(os.path.join(data_dir, str(c)))

    if randomise:
        selected_images = np.random.choice(image_paths, size=num_images_per_class)
    else:
        selected_images = image_paths[:num_images_per_class]
    for im_path in selected_images:
        im = io.imread(im_path, 'L')
    if flatten:
        im = im.flatten()

    np_data[idx] = im
    labels[idx] = c
    idx += 1

    return np_data


def convert_1d_image_to_3d(im, size=(80, 80)):
    im = data[0].reshape(1, *size)
    im = np.concatenate([im, im, im], axis=0)
    im = im.reshape(1, 3, *size)

    return im


class ResNetFeatueExtractor:
    def __init__(self):
        original_model = torchvision.models.resnet18(pretrained=True)
        self._features_layer = nn.Sequential(*list(original_model.children())[:-1])
        self._features_layer.double()
        self._device = None

    def to(self, device):
        self._device = device
        self._features_layer.to(self._device)

    def get_features_from_image_np(self, im):
        if not self._device:
            raise Exception("[Device not set]: Call instance.to(device) \
      to set approprite device from ['cpu', 'cuda']")

        im = im.astype(np.double)
        im_tensor = torch.from_numpy(im)
        im_tensor = im_tensor.to(device)
        features_vector = features(im_tensor).flatten()

        return features_vector.cpu().detach().numpy()


def get_features_from_images(im_data, feature_extractor, target_shape):
    feature_vectors = np.ones((im_data.shape[0], target_shape))

    for idx in tqdm(range(im_data.shape[0])):
        im = im_data[idx]
        im_3d = convert_1d_image_to_3d(im)
        feature_vectors[idx] = feature_extractor.get_features_from_image_np(im_3d)

    return feature_vectors


def parse_config(config_path):
    """
    Parse and return json config file
    :param config_path: str path of the config file
    :return: json object with configurations
    """
    if not os.path.isfile(config_path):
        raise Exception("Config file does not exists at {}".format(config_path))

    try:
        with open(config_path) as f:
            data = json.load(f)

            # read model config
            with open(data["model_config"]) as f:
                model_config = json.load(f)
            data['model_config'] = model_config

            return data
    except JSONDecodeError:
        raise Exception("Config parse error")
