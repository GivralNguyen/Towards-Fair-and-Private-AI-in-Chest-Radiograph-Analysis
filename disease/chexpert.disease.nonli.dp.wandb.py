
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
import yaml
import wandb
import gc
from sklearn.metrics import roc_curve, auc, recall_score
image_size = (224, 224)
num_classes = 14
torch.set_float32_matmul_precision('high')
DELTA = 1/76205
batch_size = 4096 #
# max_physical_batch_size = 32
num_workers = 4
MAX_GRAD_NORM = 1.2


def accuracy(preds, labels):
    return (preds == labels).mean()

def train_model(model, train_loader, val_loader, optimizer,writer, epoch, max_physical_batch_size, privacy_engine):

    model.train()
    losses = []
    preds = []
    targets = []
    # top1_acc = []
    best_val_loss = float('inf')  # Initialize with a large value  
    best_val_roc_auc = 0 
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
            losses.append(loss.item())
            preds.append(prob)
            targets.append(labels)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix({'Loss': loss.item()})

        preds = torch.cat(preds, dim=0).cpu().detach().numpy()
        targets = np.array(torch.cat(targets, dim=0).cpu().detach().numpy())
        target_fpr = 0.2
        fpr, tpr, thres = roc_curve(targets, preds)
        roc_auc = auc(fpr, tpr)
        op = thres[np.argmin(np.abs(fpr-target_fpr))]
        fpr_t = 1 - recall_score(targets, preds>=op, pos_label=0)
        tpr_t = recall_score(targets, preds>=op, pos_label=1)
        epsilon = privacy_engine.get_epsilon(DELTA)
        print(
            f"\tTrain Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"roc_auc = {roc_auc:.4f})"
            f"fpr_t = {fpr_t:.4f})"
            f"tpr_t = {tpr_t:.4f})"
        )   
    wandb.log({"Train/Loss": np.mean(losses), "epoch": epoch})  
    wandb.log({"roc_auc": roc_auc, "epoch": epoch}) 
    wandb.log({"fpr_t": fpr_t, "epoch": epoch}) 
    wandb.log({"tpr_t": tpr_t, "epoch": epoch})      
    wandb.log({"Epsilon": epsilon, "epoch": epoch})   
    writer.add_scalar('Train/Loss', np.mean(losses), epoch)
    print("before releasing train data and label")
    print(torch.cuda.memory_summary())
    del input_data, labels, losses, preds, targets, fpr, tpr, thres, roc_auc, op, fpr_t, tpr_t
    torch.cuda.empty_cache()
    gc.collect()
    print("after releasing train data and label")
    print(torch.cuda.memory_summary())
    model.eval()
    val_losses = []
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_data, labels = batch['image'].to('cuda'), batch['label'].to('cuda')
            output = model(input_data)
            prob = torch.sigmoid(output)
            val_loss_ = (F.binary_cross_entropy(prob, labels))
            val_losses.append(val_loss_.item())
            val_preds.append(prob)
            val_targets.append(labels)
    val_loss = np.mean(val_losses)
    val_preds = torch.cat(val_preds, dim=0).cpu().detach().numpy()
    val_targets = np.array(torch.cat(val_targets, dim=0).cpu().detach().numpy())
    val_target_fpr = 0.2
    val_fpr, val_tpr, val_thres = roc_curve(val_targets, val_preds)
    val_roc_auc = auc(val_fpr, val_tpr)
    val_op = val_thres[np.argmin(np.abs(val_fpr-val_target_fpr))]
    val_fpr_t = 1 - recall_score(val_targets, val_preds>=val_op, pos_label=0)
    val_tpr_t = recall_score(val_targets, val_preds>=val_op, pos_label=1)
    # epsilon = privacy_engine.get_epsilon(DELTA)
    print(
        f"\tVal Epoch: {epoch} \t"
        f"val_loss: {val_loss:.6f} "
        f"val_roc_auc = {val_roc_auc:.4f})"
        f"val_fpr_t = {val_fpr_t:.4f})"
        f"val_tpr_t = {val_tpr_t:.4f})"
    )   
    # Save checkpoint if validation loss improves
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), f'{writer.log_dir}/best_model.pth')
    #     print(
    #         f"Save model."
    #         f"\tVal Epoch: {epoch} \t"
    #         f"val_loss: {val_loss:.6f} "
    #         f"val_roc_auc = {val_roc_auc:.4f})"
    #         f"val_fpr_t = {val_fpr_t:.4f})"
    #         f"val_tpr_t = {val_tpr_t:.4f})"
    #     )
    if val_roc_auc > best_val_roc_auc:
        best_val_roc_auc = val_roc_auc
        torch.save(model.state_dict(), f'{writer.log_dir}/best_model.pth')
        print(
            f"Save model."
            f"\tVal Epoch: {epoch} \t"
            f"val_loss: {val_loss:.6f} "
            f"val_roc_auc = {val_roc_auc:.4f})"
            f"val_fpr_t = {val_fpr_t:.4f})"
            f"val_tpr_t = {val_tpr_t:.4f})"
        )
    wandb.log({"val_loss": val_loss, "epoch": epoch})   
    wandb.log({"val_roc_auc": val_roc_auc, "epoch": epoch}) 
    wandb.log({"val_fpr_t": val_fpr_t, "epoch": epoch}) 
    wandb.log({"val_tpr_t": val_tpr_t, "epoch": epoch})      
    writer.add_scalar('val_loss', val_loss, epoch)
    print("before releasing val data and label")
    print(torch.cuda.memory_summary())
    del input_data, labels, val_losses , val_preds, val_targets, val_fpr, val_tpr, val_thres, val_roc_auc, val_op, val_fpr_t, val_tpr_t
    torch.cuda.empty_cache()
    gc.collect()
    print("after releasing val data and label")
    print(torch.cuda.memory_summary())
    gc.collect()


def main(config=None):
    # Initialize a new wandb run
    privacy_engine = PrivacyEngine()
    with wandb.init(config=config):
        config = wandb.config
        # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
        pl.seed_everything(config.seed, workers=True)
        epochs = config.epochs
        learning_rate = config.learning_rate
        EPSILON = config.EPSILON
        # data
        data = CheXpertDataModule(csv_train_img='../data/chexpert.sample.train.csv',
                                csv_val_img='../data/chexpert.sample.val.csv',
                                csv_test_img='../data/chexpert.sample.test.csv',
                                csv_test_img_resample='../data/chexpert.resample.test.csv',
                                image_size=image_size,
                                pseudo_rgb=True,
                                batch_size=batch_size,
                                max_physical_batch_size=config.max_physical_batch_size,
                                num_workers=num_workers,
                                train_aug=False)
        # model
        out_name = config.out_name
        out_dir = f'{config.save_root}batch_{batch_size}_epochs_{epochs}_lr_{learning_rate}_target_epsilon_{EPSILON}_target delta_{DELTA}_max grad norm_{MAX_GRAD_NORM}_{out_name}'
        
        # Function to create a new directory with a suffix if the directory already exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            ver = 0
            new_dir_path = f"{out_dir}_ver{ver}"
            while os.path.exists(new_dir_path):
                ver += 1
                new_dir_path = f"{out_dir}_ver{ver}"
            os.makedirs(new_dir_path)
            
            out_dir = new_dir_path
            print("creating new save dir "+str(out_dir))
        temp_dir = os.path.join(out_dir, 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        for idx in range(0,5):
            sample = data.train_set.get_sample(idx)
            imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))
        model_type = NonLiResNetDP
        writer= SummaryWriter(log_dir=out_dir)
        model = model_type(num_classes=num_classes,epochs = epochs, writer=writer,learning_rate=learning_rate)
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
            train_model(model, train_loader,data.val_dataloader() ,optimizer=optimizer, writer=writer, epoch = epoch + 1, max_physical_batch_size = config.max_physical_batch_size, privacy_engine=privacy_engine)
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

        model._module.remove_head()

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

        # cleanup
        print("before releasing memory")
        print(torch.cuda.memory_summary())
        del model, optimizer, train_loader, data, privacy_engine, loaded_state_dict
        del preds_val, targets_val, logits_val
        del preds_test, targets_test, logits_test
        del preds_test_resample, targets_test_resample, logits_test_resample
        del embeds_val, embeds_test, embeds_test_resample
        del new_state_dict, new_state_dict_without_fc, pretrained_resnet18
        torch.cuda.empty_cache()
        gc.collect()
        print("after releasing memory")
        print(torch.cuda.memory_summary())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        sweep_config = yaml.load(f, Loader=yaml.SafeLoader)
    import pprint
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=os.path.basename(args.config))
    wandb.agent(sweep_id, main)
    wandb.finish()
    # main()  