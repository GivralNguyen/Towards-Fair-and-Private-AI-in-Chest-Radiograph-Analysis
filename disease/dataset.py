import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
img_data_dir = '/vol/aimspace/projects/CheXpert/CheXpert/'
import torch
from skimage.io import imread, imsave
from tqdm import tqdm
import pytorch_lightning as pl

class CheXpertDataset(Dataset):
    def __init__(self, csv_file_img, image_size, augmentation=False, pseudo_rgb = True):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb

        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
            T.RandomApply(transforms=[T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
            T.RandomApply(transforms=[T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = img_data_dir + self.data.loc[idx, 'path_preproc']
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label = torch.from_numpy(sample['label'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label': sample['label']}


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_val_img_resample, csv_test_img, csv_test_img_resample, image_size, pseudo_rgb, batch_size, max_physical_batch_size, num_workers, train_aug):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_val_img_resample = csv_val_img_resample
        self.csv_test_img = csv_test_img
        self.csv_test_img_resample = csv_test_img_resample
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_physical_batch_size = max_physical_batch_size
        self.num_workers = num_workers
        self.train_aug = train_aug
        self.train_set = CheXpertDataset(self.csv_train_img, self.image_size, augmentation=self.train_aug, pseudo_rgb=pseudo_rgb)
        self.val_set = CheXpertDataset(self.csv_val_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.val_set_resample = CheXpertDataset(self.csv_val_img_resample, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set = CheXpertDataset(self.csv_test_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set_resample = CheXpertDataset(self.csv_test_img_resample, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))
        print('#test resample:  ', len(self.test_set_resample))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.max_physical_batch_size, shuffle=False, num_workers=self.num_workers)

    def val_resample_dataloader(self):
        return DataLoader(self.val_set_resample, self.max_physical_batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, self.max_physical_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_resample_dataloader(self):
        return DataLoader(self.test_set_resample, self.max_physical_batch_size, shuffle=False, num_workers=self.num_workers)

