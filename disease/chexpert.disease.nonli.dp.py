
import torch
import os
from skimage.io import imsave
from dataset import CheXpertDataModule
from groupnormresnet import resnet18gn
from utils import test,freeze_model, embeddings
from model import NonLiResNetDP
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from argparse import ArgumentParser
import numpy as np
from opacus import PrivacyEngine
from torchvision import models
from tqdm import tqdm
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager
image_size = (224, 224)
num_classes = 14
batch_size = 4096
epochs = 20
max_physical_batch_size = 256
num_workers = 4
img_data_dir = '/vol/aimspace/projects/CheXpert/CheXpert/'
torch.set_float32_matmul_precision('high')
MAX_GRAD_NORM = 1.2
EPSILON = 4.0
DELTA = 1/80000
privacy_engine = PrivacyEngine()

def accuracy(preds, labels):
    return (preds == labels).mean()

def train_model(model, train_loader, val_loader, optimizer,writer, epoch):

    model.train()
    losses = []
    top1_acc = []
    best_val_loss = float('inf')  # Initialize with a large value  
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=max_physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
    # with train_loader as memory_safe_data_loader:
        train_bar = tqdm(memory_safe_data_loader, desc='Training', unit='batch')
        for batch in train_bar:
            optimizer.zero_grad()
            input_data, labels = batch['image'].to('cuda'), batch['label'].to('cuda')
            output = model(input_data)
            prob = torch.sigmoid(output)
            loss = F.binary_cross_entropy(prob, labels)
            # preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            # print(preds,labels)
            # acc = accuracy(preds, labels)
            losses.append(loss.item())
            # top1_acc.append(acc)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix({'Loss': loss.item()})
        epsilon = privacy_engine.get_epsilon(DELTA)
        print(
            f"\tTrain Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            # f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
            f"(ε = {epsilon:.2f}, δ = {DELTA})"
        )
    writer.add_scalar('Train/Loss', np.mean(losses), epoch)

    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            input_data, labels = batch['image'].to('cuda'), batch['label'].to('cuda')
            output = model(input_data)
            prob = torch.sigmoid(output)
            val_loss_ = (F.binary_cross_entropy(prob, labels))
            val_losses.append(val_loss_.item())
    val_loss = np.mean(val_losses)
    # Save checkpoint if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'{writer.log_dir}/best_model.pth')
    writer.add_scalar('Validation/Loss', val_loss, epoch)


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
                              num_workers=num_workers,
                              train_aug=False)
    # model
    out_name = 'nonli-resnetdp-all'
    out_dir = 'chexpert/diseasedp/' + out_name
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))
    model_type = NonLiResNetDP
    writer= SummaryWriter(log_dir=f'chexpert/diseasedp/batch_{batch_size}_epochs_{epochs}_target_epsilon_{EPSILON}_target delta_{DELTA}_max grad norm_{MAX_GRAD_NORM}_{out_name}')
    model = model_type(num_classes=num_classes,epochs = epochs, writer=writer)
    # Load pre-trained weights from the original ResNet-18
    pretrained_resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    new_state_dict = {}
    for key, value in pretrained_resnet18.state_dict().items():
        new_key = "model." + key  # You may need to adjust this based on your model's structure
        new_state_dict[new_key] = value
    new_state_dict_without_fc = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
    model.load_state_dict(new_state_dict_without_fc, strict=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else "cpu"

    model.to(device) 
    optimizer = model.configure_optimizers()
    
    train_loader = data.train_dataloader()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=EPSILON ,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )

    print(f"Using sigma={optimizer.noise_multiplier}, config = {writer.log_dir}")
    for epoch in tqdm(range(epochs), desc="Epoch", unit="epoch"):
        train_model(model, train_loader,data.val_dataloader() ,optimizer=optimizer, writer=writer, epoch = epoch + 1)
    # train_model(model,train_loader,data.val_dataloader() ,optimizer=optimizer, writer=writer)
    loaded_state_dict = torch.load(f'{writer.log_dir}/best_model.pth')
    # new_state_dict = {}
    # for key, value in loaded_state_dict.items():
    #     new_key = 'model.' + key
    #     new_state_dict[new_key] = value
    model.load_state_dict(loaded_state_dict)

    

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

    # model.remove_head()

    # embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    # df = pd.DataFrame(data=embeds_val)
    # df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    # df = pd.concat([df, df_targets], axis=1)
    # df.to_csv(os.path.join(writer.log_dir, 'embeddings.val.csv'), index=False)

    # embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    # df = pd.DataFrame(data=embeds_test)
    # df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    # df = pd.concat([df, df_targets], axis=1)
    # df.to_csv(os.path.join(writer.log_dir, 'embeddings.test.csv'), index=False)

    # embeds_test_resample, targets_test_resample = embeddings(model, data.test_resample_dataloader(), device)
    # df = pd.DataFrame(data=embeds_test_resample)
    # df_targets = pd.DataFrame(data=targets_test_resample, columns=cols_names_targets)
    # df = pd.concat([df, df_targets], axis=1)
    # df.to_csv(os.path.join(writer.log_dir, 'embeddings.resample.test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--accelerator', default="gpu")
    args = parser.parse_args()

    main(args)