
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
import sys
sys.path.append("/vol/aimspace/users/ngq/Towards-Fair-and-Private-AI-in-Chest-Radiograph-Analysis")
from metrics.metric_analysis import subgroup_fairness_analysis_train

image_size = (224, 224)
num_classes = 14
cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]
torch.set_float32_matmul_precision('high')
DELTA = 1/76205
batch_size = 4096 #
# max_physical_batch_size = 32
num_workers = 4
MAX_GRAD_NORM = 1.2

def accuracy(preds, labels):
    return (preds == labels).mean()

def train_model(model, train_loader, val_loader, optimizer,writer, epoch, max_physical_batch_size, privacy_engine, save_metric, best_metric):

    model.train()
    losses = []
    logits = []
    preds = []
    targets = []
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
            logits.append(output)
            preds.append(prob)
            targets.append(labels)
            loss.backward()
            optimizer.step()
            train_bar.set_postfix({'Loss': loss.item()})
    epsilon = privacy_engine.get_epsilon(DELTA)
    logits = torch.cat(logits, dim=0).detach().cpu().numpy()
    preds = torch.cat(preds, dim=0).detach().cpu().numpy()
    targets = torch.cat(targets, dim=0).detach().cpu().numpy()
    df = pd.DataFrame(data=preds, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    metrics = subgroup_fairness_analysis_train(0,"/vol/aimspace/users/ngq/Towards-Fair-and-Private-AI-in-Chest-Radiograph-Analysis/data/chexpert.sample.train.csv",df,False)
    # Define the groups and metrics to be printed
    groups = ['All', 'White', 'Asian', 'Black', 'Female', 'Male']
    metrics_to_print = ['TPR', 'FPR', 'AUC', 'AP']

    # Print the training metrics
    print(f"\tTrain Epoch: {epoch} \tTrain Loss: {np.mean(losses):.6f} (ε = {epsilon:.2f}, δ = {DELTA})", end=' ')
    for group in groups:
        print(f"({group})", end=' ')
        for metric in metrics_to_print:
            print(f"{metric} = {metrics[group][metric]:.4f}", end=' ')
    print()  # Newline at the end of the print statement
    # Log metrics to wandb
    wandb.log({
        "train_loss": np.mean(losses),
        "Epsilon": epsilon,
        "epoch": epoch,
    })

    # Log demographic-specific metrics
    for group in ['All', 'White', 'Asian', 'Black', 'Female', 'Male']:
        wandb.log({
            f"TPR_{group}": metrics[group]['TPR'],
            f"FPR_{group}": metrics[group]['FPR'],
            f"AUC_{group}": metrics[group]['AUC'],
            f"AP_{group}": metrics[group]['AP'],
            "epoch": epoch,
        })
    writer.add_scalar('Train/Loss', np.mean(losses), epoch)
    # print("before releasing train data and label")
    # print(torch.cuda.memory_summary())
    del df, df_logits, df_targets
    del metrics
    del logits, preds, targets
    del input_data, labels
    torch.cuda.empty_cache()
    gc.collect()
    # print("after releasing train data, metrics and label")
    # print(torch.cuda.memory_summary())
    model.eval()
    val_losses = []
    val_logits = []
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            input_data, labels = batch['image'].to('cuda'), batch['label'].to('cuda')
            output = model(input_data)
            prob = torch.sigmoid(output)
            val_loss_ = (F.binary_cross_entropy(prob, labels))
            val_losses.append(val_loss_.item())
            val_logits.append(output)
            val_preds.append(prob)
            val_targets.append(labels)
    val_loss = np.mean(val_losses)
    val_logits = torch.cat(val_logits, dim=0).detach().cpu().numpy()
    val_preds = torch.cat(val_preds, dim=0).detach().cpu().numpy()
    val_targets = torch.cat(val_targets, dim=0).detach().cpu().numpy()

    val_df = pd.DataFrame(data=val_preds, columns=cols_names_classes)
    val_df_logits = pd.DataFrame(data=val_logits, columns=cols_names_logits)
    val_df_targets = pd.DataFrame(data=val_targets, columns=cols_names_targets)
    val_df = pd.concat([val_df, val_df_logits, val_df_targets], axis=1)
    val_metrics = subgroup_fairness_analysis_train(0,"/vol/aimspace/users/ngq/Towards-Fair-and-Private-AI-in-Chest-Radiograph-Analysis/data/chexpert.sample.val.csv",val_df,False)
    # Print the validation metrics
    print(f"\Val Epoch: {epoch} \Val Loss: {val_loss:.6f}", end=' ')
    for group in groups:
        print(f"({group})", end=' ')
        for metric in metrics_to_print:
            print(f"{metric} = {val_metrics[group][metric]:.4f}", end=' ')
    print()  # Newline at the end of the print statement
    # Log metrics to wandb
    wandb.log({
        "val_loss": val_loss,
        "epoch": epoch,
    })

    # Log demographic-specific metrics
    for group in ['All', 'White', 'Asian', 'Black', 'Female', 'Male']:
        wandb.log({
            f"val_TPR_{group}": val_metrics[group]['TPR'],
            f"val_FPR_{group}": val_metrics[group]['FPR'],
            f"val_AUC_{group}": val_metrics[group]['AUC'],
            f"val_AP_{group}": val_metrics[group]['AP'],
            "epoch": epoch,
        })
    # Save checkpoint if validation loss improves
    if save_metric == 'minimize_val_loss':
        best_val_loss = best_metric
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{writer.log_dir}/best_model.pth')
            print(
                f"Save model. Best Val Loss: {val_loss:.6f} "
            )
        best_metric = best_val_loss
    elif save_metric == 'maximize_val_AUC_All':
        best_val_roc_auc = best_metric
        if val_metrics['All']['AUC']  > best_val_roc_auc:
            best_val_roc_auc = val_metrics['All']['AUC'] 
            torch.save(model.state_dict(), f'{writer.log_dir}/best_model.pth')
            print(
                f"Save model."
                f"Best val_roc_auc = {val_metrics['All']['AUC'] :.4f})"
            )
        best_metric = best_val_roc_auc
    else:
        raise Exception("Invalid save_metric specified")
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('val_roc_auc_all', val_metrics['All']['AUC'], epoch)
    # print("before releasing val data and label")
    # print(torch.cuda.memory_summary())
    del input_data, labels
    del val_df, val_df_logits, val_df_targets
    del val_metrics, val_loss
    del val_logits, val_preds, val_targets
    torch.cuda.empty_cache()
    gc.collect()
    # print("after releasing val data, metrics and label")
    # print(torch.cuda.memory_summary())
    return best_metric
    


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
        # modelminimize_val_loss
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
        if config.save_metric == 'minimize_val_loss':
            best_metric = float('inf')  
        elif  config.save_metric == 'maximize_val_AUC_All':
            best_metric = 0 
        else:
            raise Exception("Invalid save_metric specified")
        for epoch in tqdm(range(epochs), desc="Epoch", unit="epoch"):
            best_metric = train_model(model, train_loader,data.val_dataloader() ,optimizer=optimizer, writer=writer, epoch = epoch + 1, max_physical_batch_size = config.max_physical_batch_size, privacy_engine=privacy_engine, save_metric = config.save_metric, best_metric = best_metric)
        # train_model(model,train_loader,data.val_dataloader() ,optimizer=optimizer, writer=writer)
        loaded_state_dict = torch.load(f'{writer.log_dir}/best_model.pth')
        # new_state_dict = {}
        # for key, value in loaded_state_dict.items():
        #     new_key = 'model.' + key
        #     new_state_dict[new_key] = value
        model.load_state_dict(loaded_state_dict)

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
        # print("before releasing memory")
        # print(torch.cuda.memory_summary())
        del model, optimizer, train_loader, data, privacy_engine, loaded_state_dict
        del preds_val, targets_val, logits_val
        del preds_test, targets_test, logits_test
        del preds_test_resample, targets_test_resample, logits_test_resample
        del embeds_val, embeds_test, embeds_test_resample
        del new_state_dict, new_state_dict_without_fc, pretrained_resnet18
        torch.cuda.empty_cache()
        gc.collect()
        # print("after releasing memory")
        # print(torch.cuda.memory_summary())


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