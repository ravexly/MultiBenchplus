import torch
from torch.utils.data import Dataset
from spiketransform import transformRate
import torch.utils.data as data
import numpy as np
import scipy.io as sio
import h5py
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text



class CustomTensorDataset_clip(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x_img = self.tensors[0][index]

        if self.transform:
            x_img = self.transform(x_img)

        x_audio = self.tensors[1][index]

        y_img = self.tensors[2][index]
        y_audio = self.tensors[3][index]


        return x_img, x_audio, y_img

    def __len__(self):
        return self.tensors[0].size(0)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    

class SpikingEMNIST(Dataset):
    def __init__(self, emnist_data, N_ts, max_is_present_for, seed=0):
        self.emnist_data = emnist_data
        self.N_ts = N_ts
        self.max_is_present_for = max_is_present_for
        self.seed = seed

    def __len__(self):
        return len(self.emnist_data)

    def __getitem__(self, index):
        data, target = self.emnist_data[index]
        spiking_data = torch.from_numpy(transformRate(data, self.N_ts, self.max_is_present_for, seed=self.seed)).float()  # Convert to float
        return spiking_data, target
    
    
def filter_first_n_letters(data, n=10):
    filtered_data = []
    for i in range(len(data)):
        img, label = data[i]
        if 1 <= label <= n:
            filtered_data.append((img, label))
    return filtered_data



def sum_data(data_list, indices):
    result = data_list[indices[0]]
    for i in indices[1:]:
        result += data_list[i]
    return result
class CustomTensorDataset_pro(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None, stratified=True):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.stratified = stratified

        y_sup_img = self.tensors[2]
        y_sup_aud = self.tensors[3]
        self.img_cls_idx, self.aud_cls_idx = [], []
        for i in torch.unique(y_sup_img):
            img_idx = torch.where(y_sup_img == i)[0]
            aud_idx = torch.where(y_sup_aud == i)[0]
            self.img_cls_idx.append(img_idx)
            self.aud_cls_idx.append(aud_idx)

    def __getitem__(self, index):
        x_img = self.tensors[0][index]

        if self.transform:
            x_img = self.transform(x_img)

        x_audio = self.tensors[1][index]

        y_img = self.tensors[2][index]
        y_audio = self.tensors[3][index]

        if len(self.tensors) == 8:
            x_que_img = self.tensors[4][index]
            x_que_aud = self.tensors[5][index]
            y_que_img = self.tensors[6][index]
            y_que_aud = self.tensors[7][index]

        return x_img, x_audio, y_img, y_audio, x_que_img, x_que_aud, y_que_img, y_que_aud

    def __len__(self):
        return self.tensors[0].size(0)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x_img = self.tensors[0][index]

        if self.transform:
            x_img = self.transform(x_img)

        x_audio = self.tensors[1][index]

        y_img = self.tensors[2][index]
        y_audio = self.tensors[3][index]


        return x_img, x_audio, y_img, y_audio

    def __len__(self):
        return self.tensors[0].size(0)

class MyDataset(data.Dataset):
    def __init__(self, path='load_test.mat', method='h', lens=15):
        if method == 'h':
            data = h5py.File(path)
            image, label = data['image'], data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :, :]
            self.labels = torch.from_numpy(label).float()

        elif method == 'nmnist_r':
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()
            self.images = self.images.permute(0, 3, 1, 2, 4)


        elif method == 'nmnist_h':
            data = h5py.File(path)
            image, label = data['image'], data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :, :]
            self.labels = torch.from_numpy(label).float()
            self.images = self.images.permute(0, 3, 1, 2, 4)

        else:
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()
        self.num_sample = int((len(self.images) // 100) * 100)
        print(self.images.size(), self.labels.size())

    def __getitem__(self, index):  # 返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample


class AudioDataset(data.Dataset):
    def __init__(self, data_path='./data/trainX_4ms.npy', label_path='./data/trainY_4ms.npy', lens=2):
        #         data = sio.loadmat(path)
        data = np.load(data_path)
        label = np.load(label_path)
        self.images = torch.from_numpy(data)
        self.labels = torch.from_numpy(label.astype(float)).float()
        #         print(self.labels.shape)
        #         one_hot = torch.zeros(self.labels.shape[0], 20)
        #         one_hot[range(self.labels.shape[0]), self.labels]=1
        self.images = self.images.permute(0, 2, 1)

        self.num_sample = int((len(self.images) // 100) * 100)
        print(self.images.size(), self.labels.size())

    def __getitem__(self, index):  # 返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample


class MyCenterCropDataset(data.Dataset):
    def __init__(self, path='load_test.mat', method='h', new_width=20, new_height=20, lens=15):
        if method == 'h':
            data = h5py.File(path)
            image, label = data['image'], data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :, :]
            self.labels = torch.from_numpy(label).float()

        elif method == 'nmnist_r':
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()

            self.new_width = new_width
            self.new_height = new_height
            width = self.images.shape[2]
            height = self.images.shape[1]
            # print('Orignal width is ', width)
            # print('Orignal height is', height)
            left = int(np.ceil((width - self.new_width) / 2))
            right = width - int(np.floor((width - self.new_width) / 2))
            top = int(np.ceil((height - self.new_height) / 2))
            bottom = height - int(np.floor((height - self.new_height) / 2))

            self.images = self.images[:, top:bottom, left:right, :, :]

            self.images = self.images.permute(0, 3, 1, 2, 4)


        elif method == 'nmnist_h':
            data = h5py.File(path)
            image, label = data['image'], data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)

            self.new_width = new_width
            self.new_height = new_height
            width = image.shape[2]
            height = image.shape[1]
            # print('Orignal width is ', width)
            # print('Orignal height is', height)
            left = int(np.ceil((width - self.new_width) / 2))
            right = width - int(np.floor((width - self.new_width) / 2))
            top = int(np.ceil((height - self.new_height) / 2))
            bottom = height - int(np.floor((height - self.new_height) / 2))

            self.images = self.images[:, top:bottom, left:right, :, :]

            self.labels = torch.from_numpy(label).float()
            self.images = self.images.permute(0, 3, 1, 2, 4)

        else:
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()
        self.num_sample = int((len(self.images) // 100) * 100)
        # print(self.images.size(),self.labels.size())

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample