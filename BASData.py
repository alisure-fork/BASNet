import math
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform, color


class RescaleT(object):

    def __init__(self, output_size):
        self.output_size = output_size
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), 0, mode='constant', preserve_range=True)
        return {'image': img, 'label': lbl}

    pass


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h = self.output_size * h / w if h > w else self.output_size
        new_w = self.output_size if h > w else self.output_size * w / h
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)
        return {'image': img, 'label': lbl}

    pass


class CenterCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size
        h_offset = int(math.floor((h - new_h) / 2))
        w_offset = int(math.floor((w - new_w) / 2))

        image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
        label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
        return {'image': image, 'label': label}

    pass


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        return {'image': image, 'label': label}

    pass


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        image = image / np.max(image)
        label = label if np.max(label) < 1e-6 else (label / np.max(label))

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
            pass

        tmpLbl[:, :, 0] = label[:, :, 0]
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg),  'label': torch.from_numpy(tmpLbl)}

    pass


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        label_3 = np.zeros(image.shape) if 0 == len(self.label_name_list) else io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3
            pass

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]
            pass

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    pass
