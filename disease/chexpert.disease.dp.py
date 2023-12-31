import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.lightning import DPLightningDataModule
from opacus.utils.batch_memory_manager import BatchMemoryManager


image_size = (224, 224)
num_classes = 14
# batch_size = 1024
# max_physical_batch_size = 256
# MAX_GRAD_NORM = 1.2
# EPSILON = 50.0
# DELTA = 1/80000
epochs = 20
num_workers = 4
img_data_dir = '/vol/aimspace/projects/CheXpert/CheXpert/'
torch.set_float32_matmul_precision('high')

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
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, csv_test_img_resample, image_size, pseudo_rgb, batch_size, max_physical_batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.csv_test_img_resample = csv_test_img_resample
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_physical_batch_size = max_physical_batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(self.csv_train_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.val_set = CheXpertDataset(self.csv_val_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
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

    def test_dataloader(self):
        return DataLoader(self.test_set, self.max_physical_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_resample_dataloader(self):
        return DataLoader(self.test_set_resample, self.max_physical_batch_size, shuffle=False, num_workers=self.num_workers)


from groupnormresnet import resnet18gn

# Instantiate the custom model


class ResNetDP(pl.LightningModule):
    def __init__(self, num_classes,
        enable_dp: bool = True,  
        epochs=epochs,
        target_epsilon=50.0,
        target_delta=1/80000,
        max_grad_norm=1.2,
        noise_multiplier: float = 1.0,
        ):
        """A Resnet for classifying with differential privacy training
        Args:
            lr: Learning rate
            enable_dp: Enables training with privacy guarantees using Opacus (if True), vanilla SGD otherwise
            delta: Target delta for which (eps, delta)-DP is computed
            noise_multiplier: Noise multiplier
            max_grad_norm: Clip per-sample gradients to this norm
        """
        super().__init__()
        self.num_classes = num_classes
        self.model = resnet18gn()
        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        self.enable_dp = enable_dp
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        if self.enable_dp:
            self.privacy_engine = PrivacyEngine()
    def remove_head(self):
        num_features = self.model.fc.in_features
        id_layer = nn.Identity(num_features)
        self.model.fc = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0)

        if self.enable_dp:
            self.trainer.fit_loop.setup_data()
            data_loader = self.trainer.train_dataloader
            
            # transform (model, optimizer, dataloader) to DP-versions
            if hasattr(self, "dp"):
                self.dp["model"].remove_hooks()
            dp_model, optimizer, dataloader = self.privacy_engine.make_private_with_epsilon(
                module=self,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=self.epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_grad_norm,
            )
            

            self.dp = {"model": dp_model}
            print(f"Using sigma={optimizer.noise_multiplier}, batch size = {batch_size},  epochs = {self.epochs}, target_epsilon ={self.target_epsilon}, target delta = {self.target_delta} ,  max grad norm={self.max_grad_norm}")
        # params_to_update = []
        # for param in self.parameters():
        #     if param.requires_grad == True:
        #         params_to_update.append(param)
        # optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)

    def on_train_epoch_end(self):
        # Logging privacy spent: (epsilon, delta)
        epsilon = self.privacy_engine.get_epsilon(self.target_delta)
        self.log("epsilon", epsilon, on_epoch=True, prog_bar=True)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []
    
    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()


def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    # data
    data = CheXpertDataModule(csv_train_img='../data/chexpert.sample.train.csv',
                              csv_val_img='../data/chexpert.sample.val.csv',
                              csv_test_img='../data/chexpert.sample.test.csv',
                              csv_test_img_resample='../data/chexpert.resample.test.csv',
                              image_size=image_size,
                              pseudo_rgb=True,
                              batch_size=batch_size,
                              max_physical_batch_size=max_physical_batch_size,
                              num_workers=num_workers)

    # model
    model_type = ResNetDP
    model = model_type(target_epsilon=hparams.epsilon,target_delta=hparams.delta,max_grad_norm=hparams.max_grad_norm,num_classes=num_classes)
    # Load pre-trained weights from the original ResNet-18
    pretrained_resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    new_state_dict = {}
    for key, value in pretrained_resnet18.state_dict().items():
        new_key = "model." + key  # You may need to adjust this based on your model's structure
        new_state_dict[new_key] = value
    new_state_dict_without_fc = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
    model.load_state_dict(new_state_dict_without_fc, strict=False)

    # from opacus.validators import ModuleValidator

    # errors = ModuleValidator.validate(model, strict=False)
    # errors[-5:]
    # model = ModuleValidator.fix(model)
    # ModuleValidator.validate(model, strict=False)
    # Create output directory
    out_name = 'resnet-all'
    out_dir = f'chexpert/disease_dp/batch_{batch_size}_physbatch_{max_physical_batch_size}_gradnorm_{MAX_GRAD_NORM}_epsilon_{EPSILON}_delta_{DELTA}_epochs_{epochs}/{out_name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=epochs,
        accelerator=hparams.accelerator,
        logger=TensorBoardLogger(f'chexpert/disease_dp/batch_{batch_size}_physbatch_{max_physical_batch_size}_gradnorm_{MAX_GRAD_NORM}_epsilon_{EPSILON}_delta_{DELTA}_epochs_{epochs}', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    dp_data = DPLightningDataModule(data)
    trainer.fit(model, dp_data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else "cpu"

    model.to(device)

    cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.val.csv'), index=False)

    print('TESTING')
    preds_test, targets_test, logits_test = test(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.csv'), index=False)

    print('TESTING RESAMPLE')
    preds_test_resample, targets_test_resample, logits_test_resample = test(model, data.test_resample_dataloader(), device)
    df = pd.DataFrame(data=preds_test_resample, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test_resample, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test_resample, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.resample.test.csv'), index=False)

    print('EMBEDDINGS')

    model.remove_head()

    embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.val.csv'), index=False)

    embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.test.csv'), index=False)

    embeds_test_resample, targets_test_resample = embeddings(model, data.test_resample_dataloader(), device)
    df = pd.DataFrame(data=embeds_test_resample)
    df_targets = pd.DataFrame(data=targets_test_resample, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.resample.test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--accelerator', default="cpu")
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--max_physical_batch_size', type=int, default=256, help='Maximum physical batch size')
    parser.add_argument('--max_grad_norm', type=float, default=1.2, help='Maximum gradient norm')
    parser.add_argument('--epsilon', type=float, default=50.0, help='Epsilon value')
    parser.add_argument('--delta', type=float, default=1/80000, help='Delta value')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')

    args = parser.parse_args()

    # Now you can access the command-line arguments as attributes of the args object
    batch_size = args.batch_size
    max_physical_batch_size = args.max_physical_batch_size
    MAX_GRAD_NORM = args.max_grad_norm
    EPSILON = args.epsilon
    DELTA = args.delta
    epochs = args.epochs

    main(args)