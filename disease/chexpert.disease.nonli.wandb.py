
import torch
import os
from skimage.io import imsave
from dataset import CheXpertDataModule
from groupnormresnet import resnet18gn
from utils import test,freeze_model, embeddings
from model import NonLiResNet
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
import copy
sys.path.append("/vol/aimspace/users/ngq/Towards-Fair-and-Private-AI-in-Chest-Radiograph-Analysis")
from metrics.metric_analysis import subgroup_fairness_analysis_train
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) #shut tqdm up
 
image_size = (224, 224)
num_classes = 14
cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]
torch.set_float32_matmul_precision('high')
num_workers = 4

def train_model(model, train_loader, val_loader, optimizer,writer, epoch, best_metrics):

    model.train()
    losses = []
    
    train_bar = tqdm(train_loader, desc='Training', unit='batch')
    for batch in train_bar:
        optimizer.zero_grad()
        input_data, labels = batch['image'].to('cuda'), batch['label'].to('cuda')
        output = model(input_data)
        prob = torch.sigmoid(output)
        loss = F.binary_cross_entropy(prob, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        train_bar.set_postfix({'Loss': loss.item()})

    groups = ['All', 'White', 'Asian', 'Black', 'Female', 'Male']
    metrics_to_print = ['TPR', 'FPR', 'AUC', 'AP']

    # Print the training metrics
    print(f"\tTrain Epoch: {epoch} \tTrain Loss: {np.mean(losses):.6f})", end=' ')
    # Log metrics to wandb
    wandb.log({
        "train_loss": np.mean(losses),
        "epoch": epoch,
    })

    # Log demographic-specific metrics
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
    
    # Log metrics to wandb
    wandb.log({
        "val_loss": val_loss,
        "epoch": epoch,
    })
    # Initialize variables to store TPR and AUC values
    tprs = []
    aucs = []
    # Dictionary to map subgroup names to TPR values for easy retrieval
    tpr_dict = {}

    # Iterate over subgroups to gather TPR and AUC values
    for group in ['All', 'White', 'Asian', 'Black', 'Female', 'Male']:
        tpr = val_metrics[group]['TPR']
        auc = val_metrics[group]['AUC']
        tprs.append(tpr)
        aucs.append(auc)
        tpr_dict[group] = tpr

    # Calculate the largest TPR gap between subgroups
    largest_tpr_gap = max(tprs) - min(tprs)

    # Find the worst subgroup TPR
    worst_tpr = min(tprs)
    worst_tpr_subgroup = min(tpr_dict, key=tpr_dict.get)
    overall_AUC_minus_TPR_gap = val_metrics['All']['AUC'] - largest_tpr_gap
    # Log demographic-specific metrics
    for group in ['All', 'White', 'Asian', 'Black', 'Female', 'Male']:
        wandb.log({
            f"val_TPR_{group}": val_metrics[group]['TPR'],
            # f"val_FPR_{group}": val_metrics[group]['FPR'],
            f"val_AUC_{group}": val_metrics[group]['AUC'],
            # f"val_AP_{group}": val_metrics[group]['AP'],
            "epoch": epoch,
        })

    wandb.log({
        "largest_tpr_gap": largest_tpr_gap,
        "worst_tpr": worst_tpr,
        # "worst_tpr_subgroup": worst_tpr_subgroup,
        "overall_AUC_minus_TPR_gap": overall_AUC_minus_TPR_gap,
        "epoch": epoch,
    })

    print(f"\Val Epoch: {epoch} \Val Loss: {val_loss:.6f}", end=' ')
    # Print subgroup-specific metrics
    for group in groups:
        print(f"({group})", end=' ')
        for metric in metrics_to_print:
            print(f"{metric} = {val_metrics[group][metric]:.4f}", end=' ')

    # Print additional fairness metrics
    print("\n")
    print(f"Largest TPR gap: {largest_tpr_gap:.4f}")
    print(f"Worst subgroup TPR: {worst_tpr:.4f}")
    print(f"The group with worst TPR: {worst_tpr_subgroup}")
    print(f"overall_AUC_minus_TPR_gap: {overall_AUC_minus_TPR_gap:.4f}")
    

    max_val_roc_auc = best_metrics['max_overall_AUC'] 
    if val_metrics['All']['AUC']  > max_val_roc_auc:
        max_val_roc_auc = val_metrics['All']['AUC'] 
        os.makedirs(f'{writer.log_dir}/max_overall_AUC', exist_ok=True)
        torch.save(model.state_dict(), f'{writer.log_dir}/max_overall_AUC/best_model.pth')
        print(
            f"Save model max val_roc_auc."
            f"Max val_roc_auc = {val_metrics['All']['AUC'] :.4f})"
        )
        best_metrics['max_overall_AUC']  = max_val_roc_auc

    max_worse_case_group_TPR = best_metrics['max_worse_case_group_TPR'] 
    if worst_tpr  > max_worse_case_group_TPR:
        max_worse_case_group_TPR = worst_tpr
        os.makedirs(f'{writer.log_dir}/max_worse_case_group_TPR', exist_ok=True)
        torch.save(model.state_dict(), f'{writer.log_dir}/max_worse_case_group_TPR/best_model.pth')
        print(
            f"Save model max_worse_case_group_TPR."
            f"max_worse_case_group_TPR= {worst_tpr :.4f})"
        )
        best_metrics['max_worse_case_group_TPR']  = max_worse_case_group_TPR

    min_largest_TPR_gap = best_metrics['min_largest_TPR_gap'] 
    if largest_tpr_gap  < min_largest_TPR_gap:
        min_largest_TPR_gap = largest_tpr_gap
        os.makedirs(f'{writer.log_dir}/min_largest_TPR_gap', exist_ok=True)
        torch.save(model.state_dict(), f'{writer.log_dir}/min_largest_TPR_gap/best_model.pth')
        print(
            f"Save model min_largest_TPR_gap."
            f"min_largest_TPR_gap= {largest_tpr_gap :.4f})"
        )
        best_metrics['min_largest_TPR_gap']  = min_largest_TPR_gap
    
    max_overall_AUC_minus_TPR_gap = best_metrics['max_overall_AUC_minus_TPR_gap'] 
    if overall_AUC_minus_TPR_gap  > max_overall_AUC_minus_TPR_gap:
        max_overall_AUC_minus_TPR_gap = overall_AUC_minus_TPR_gap
        os.makedirs(f'{writer.log_dir}/max_overall_AUC_minus_TPR_gap', exist_ok=True)
        torch.save(model.state_dict(), f'{writer.log_dir}/max_overall_AUC_minus_TPR_gap/best_model.pth')
        print(
            f"Save model max_overall_AUC_minus_TPR_gap"
            f"max_overall_AUC_minus_TPR_gap= {overall_AUC_minus_TPR_gap :.4f})"
        )
        best_metrics['max_overall_AUC_minus_TPR_gap']  = max_overall_AUC_minus_TPR_gap


    del input_data, labels
    del val_df, val_df_logits, val_df_targets
    del val_metrics, val_loss
    del val_logits, val_preds, val_targets
    del largest_tpr_gap, worst_tpr
    del tprs, aucs
    torch.cuda.empty_cache()
    gc.collect()
    return best_metrics
    


def main(config=None):
    # Initialize a new wandb run
    privacy_engine = PrivacyEngine()
    with wandb.init(config=config):
        config = wandb.config
        # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
        pl.seed_everything(config.seed, workers=True)
        epochs = config.epochs
        learning_rate = config.learning_rate
        # data
        data = CheXpertDataModule(csv_train_img='../data/chexpert.sample.train.csv',
                                csv_val_img='../data/chexpert.sample.val.csv',
                                csv_test_img='../data/chexpert.sample.test.csv',
                                csv_test_img_resample='../data/chexpert.resample.test.csv',
                                image_size=image_size,
                                pseudo_rgb=True,
                                batch_size=config.batch_size,
                                max_physical_batch_size=config.max_physical_batch_size,
                                num_workers=num_workers,
                                train_aug=True)
        # modelminimize_val_loss
        out_name = config.out_name
        out_dir = f'{config.save_root}batch_{config.batch_size,}_epochs_{epochs}_lr_{learning_rate}_non_private_{out_name}'
        
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
        model_type = NonLiResNet
        writer= SummaryWriter(log_dir=out_dir)
        model = model_type(num_classes=num_classes,epochs = epochs, writer=writer,learning_rate=learning_rate)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda") if use_cuda else "cpu"

        model.to(device) 
        optimizer = model.configure_optimizers()
        
        train_loader = data.train_dataloader()

        print(f"Using sigma={optimizer.noise_multiplier}, config = {writer.log_dir}")
        best_metrics = {}
        best_metrics['max_overall_AUC'] = 0.0
        best_metrics['max_overall_AUC_minus_TPR_gap'] = 0.0
        best_metrics['min_largest_TPR_gap'] = float('inf')
        best_metrics['max_worse_case_group_TPR'] = 0.0
        
        for epoch in tqdm(range(epochs), desc="Epoch", unit="epoch"):
            best_metrics = train_model(model, train_loader,data.val_dataloader() ,optimizer=optimizer, writer=writer, epoch = epoch + 1, max_physical_batch_size = config.max_physical_batch_size, privacy_engine=privacy_engine, best_metrics = best_metrics)
        # train_model(model,train_loader,data.val_dataloader() ,optimizer=optimizer, writer=writer)

        for metric_name, metric_value in best_metrics.items():
            loaded_state_dict = torch.load(f'{writer.log_dir}/{metric_name}/best_model.pth')
            model.load_state_dict(loaded_state_dict)
            print(f'{writer.log_dir}/{metric_name}/best_model.pth')
            print('VALIDATION')
            preds_val, targets_val, logits_val = test(model, data.val_dataloader(),num_classes, device)
            df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
            df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
            df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
            df = pd.concat([df, df_logits, df_targets], axis=1)
            df.to_csv(f'{writer.log_dir}/{metric_name}/predictions.val.csv', index=False)

            print('TESTING')
            preds_test, targets_test, logits_test = test(model, data.test_dataloader(), num_classes, device)
            df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
            df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
            df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
            df = pd.concat([df, df_logits, df_targets], axis=1)
            df.to_csv(f'{writer.log_dir}/{metric_name}/predictions.test.csv', index=False)

            print('TESTING RESAMPLE')
            preds_test_resample, targets_test_resample, logits_test_resample = test(model, data.test_resample_dataloader(),num_classes, device)
            df = pd.DataFrame(data=preds_test_resample, columns=cols_names_classes)
            df_logits = pd.DataFrame(data=logits_test_resample, columns=cols_names_logits)
            df_targets = pd.DataFrame(data=targets_test_resample, columns=cols_names_targets)
            df = pd.concat([df, df_logits, df_targets], axis=1)
            df.to_csv(f'{writer.log_dir}/{metric_name}/predictions.resample.test.csv', index=False)

            print('EMBEDDINGS')
            # print(model)
            head = model._module.remove_head()
            # print(model)
            embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
            df = pd.DataFrame(data=embeds_val)
            df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
            df = pd.concat([df, df_targets], axis=1)
            df.to_csv(f'{writer.log_dir}/{metric_name}/embeddings.val.csv', index=False)

            embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
            df = pd.DataFrame(data=embeds_test)
            df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
            df = pd.concat([df, df_targets], axis=1)
            df.to_csv(f'{writer.log_dir}/{metric_name}/embeddings.test.csv', index=False)

            embeds_test_resample, targets_test_resample = embeddings(model, data.test_resample_dataloader(), device)
            df = pd.DataFrame(data=embeds_test_resample)
            df_targets = pd.DataFrame(data=targets_test_resample, columns=cols_names_targets)
            df = pd.concat([df, df_targets], axis=1)
            df.to_csv(f'{writer.log_dir}/{metric_name}/embeddings.resample.test.csv', index=False)
            # print(model)
            model._module.restore_head(head)
            # print(model)
            # cleanup
            # print("before releasing memory")
            # print(torch.cuda.memory_summary())
            del preds_val, targets_val, logits_val
            del preds_test, targets_test, logits_test
            del preds_test_resample, targets_test_resample, logits_test_resample
            del embeds_val, embeds_test, embeds_test_resample
            # del new_state_dict, new_state_dict_without_fc, pretrained_resnet18
            torch.cuda.empty_cache()
            gc.collect()
            # print("after releasing memory")
            # print(torch.cuda.memory_summary())
        del model, optimizer, train_loader, data, privacy_engine, loaded_state_dict
        del new_state_dict, new_state_dict_without_fc, pretrained_resnet18
        torch.cuda.empty_cache()
        gc.collect()

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