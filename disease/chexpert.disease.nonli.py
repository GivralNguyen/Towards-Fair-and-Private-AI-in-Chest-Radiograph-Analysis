
import torch
import os
from skimage.io import imsave
from dataset import CheXpertDataModule, CheXpertDataset
from groupnormresnet import resnet18gn
from utils import test,freeze_model, embeddings
from model import NonLiResNet
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from argparse import ArgumentParser
import numpy as np
image_size = (224, 224)
num_classes = 14
batch_size = 256
epochs = 20
num_workers = 4
img_data_dir = '/vol/aimspace/projects/CheXpert/CheXpert/'
torch.set_float32_matmul_precision('high')

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
                              max_physical_batch_size=batch_size,
                              num_workers=num_workers,
                              train_aug=True)
    # model
    out_name = 'nonli-resnet-all'
    out_dir = 'chexpert/disease/' + out_name
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))
    model_type = NonLiResNet
    writer= SummaryWriter(log_dir=f'chexpert/disease/batch_{batch_size}_epochs_{epochs}_{out_name}')
    model = model_type(num_classes=num_classes,epochs = epochs, writer=writer)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else "cpu"

    model.to(device) 
    optimizer = model.configure_optimizers()
    model.train_model(data.train_dataloader(),data.val_dataloader() ,optimizer)
    loaded_state_dict = torch.load(f'{writer.log_dir}/best_model.pth')
    new_state_dict = {}
    for key, value in loaded_state_dict.items():
        new_key = 'model.' + key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test(model, data.val_dataloader(),num_classes, device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(writer.log_dir, 'predictions.val.csv'), index=False)

    print('TESTING')
    preds_test, targets_test, logits_test = test(model, data.test_dataloader(), num_classes, device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(writer.log_dir, 'predictions.test.csv'), index=False)

    print('TESTING RESAMPLE')
    preds_test_resample, targets_test_resample, logits_test_resample = test(model, data.test_resample_dataloader(),num_classes, device)
    df = pd.DataFrame(data=preds_test_resample, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test_resample, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test_resample, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(writer.log_dir, 'predictions.resample.test.csv'), index=False)

    print('EMBEDDINGS')

    model.remove_head()

    embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(writer.log_dir, 'embeddings.val.csv'), index=False)

    embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(writer.log_dir, 'embeddings.test.csv'), index=False)

    embeds_test_resample, targets_test_resample = embeddings(model, data.test_resample_dataloader(), device)
    df = pd.DataFrame(data=embeds_test_resample)
    df_targets = pd.DataFrame(data=targets_test_resample, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(writer.log_dir, 'embeddings.resample.test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--accelerator', default="gpu")
    args = parser.parse_args()

    main(args)