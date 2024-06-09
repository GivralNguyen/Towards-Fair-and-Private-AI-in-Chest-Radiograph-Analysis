import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import numpy as np
import wandb
import gc
import os
import sys
sys.path.append("../")
from metrics.metric_analysis import subgroup_fairness_analysis_train
num_classes = 14
cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test(model, data_loader,num_classes, device):
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


def process_validation(loader, model, csv_path, writer, prefix, epoch, best_metrics):
    val_losses = []
    val_logits = []
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in loader:
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
    val_metrics = subgroup_fairness_analysis_train(0,csv_path,val_df,False)

    # Log metrics to wandb
    wandb.log({
        f"{prefix}_loss": val_loss,
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
            f"{prefix}_TPR_{group}": val_metrics[group]['TPR'],
            # f"val_FPR_{group}": val_metrics[group]['FPR'],
            f"{prefix}_AUC_{group}": val_metrics[group]['AUC'],
            # f"val_AP_{group}": val_metrics[group]['AP'],
            "epoch": epoch,
        })

    wandb.log({
        f"{prefix}_largest_tpr_gap": largest_tpr_gap,
        f"{prefix}_worst_tpr": worst_tpr,
        f"{prefix}_overall_AUC_minus_TPR_gap": overall_AUC_minus_TPR_gap,
        "epoch": epoch,
    })

    groups = ['All', 'White', 'Asian', 'Black', 'Female', 'Male']
    metrics_to_print = ['TPR', 'FPR', 'AUC', 'AP']
    print(f"\n{prefix} Val Epoch: {epoch} {prefix} Val Loss: {val_loss:.6f}", end=' ')
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

    max_val_roc_auc = best_metrics[f'{prefix}_max_overall_AUC'] 
    if val_metrics['All']['AUC']  > max_val_roc_auc:
        max_val_roc_auc = val_metrics['All']['AUC'] 
        os.makedirs(f'{writer.log_dir}/{prefix}_max_overall_AUC', exist_ok=True)
        torch.save(model.state_dict(), f'{writer.log_dir}/{prefix}_max_overall_AUC/best_model.pth')
        print(
            f"Save model {prefix}_max val_roc_auc."
            f"{prefix}_Max val_roc_auc = {val_metrics['All']['AUC'] :.4f})"
        )
        best_metrics[f'{prefix}_max_overall_AUC']  = max_val_roc_auc

    max_worse_case_group_TPR = best_metrics[f'{prefix}_max_worse_case_group_TPR'] 
    if worst_tpr  > max_worse_case_group_TPR:
        max_worse_case_group_TPR = worst_tpr
        os.makedirs(f'{writer.log_dir}/{prefix}_max_worse_case_group_TPR', exist_ok=True)
        torch.save(model.state_dict(), f'{writer.log_dir}/{prefix}_max_worse_case_group_TPR/best_model.pth')
        print(
            f"Save model {prefix}_max_worse_case_group_TPR."
            f"{prefix}_max_worse_case_group_TPR= {worst_tpr :.4f})"
        )
        best_metrics[f'{prefix}_max_worse_case_group_TPR']  = max_worse_case_group_TPR

    # min_largest_TPR_gap = best_metrics[f'{prefix}_min_largest_TPR_gap'] 
    # if largest_tpr_gap  < min_largest_TPR_gap:
    #     min_largest_TPR_gap = largest_tpr_gap
    #     os.makedirs(f'{writer.log_dir}/{prefix}_min_largest_TPR_gap', exist_ok=True)
    #     torch.save(model.state_dict(), f'{writer.log_dir}/{prefix}_min_largest_TPR_gap/best_model.pth')
    #     print(
    #         f"Save model {prefix}_min_largest_TPR_gap."
    #         f"{prefix}_min_largest_TPR_gap= {largest_tpr_gap :.4f})"
    #     )
    #     best_metrics[f'{prefix}_min_largest_TPR_gap']  = min_largest_TPR_gap
    
    max_overall_AUC_minus_TPR_gap = best_metrics[f'{prefix}_max_overall_AUC_minus_TPR_gap'] 
    if overall_AUC_minus_TPR_gap  > max_overall_AUC_minus_TPR_gap:
        max_overall_AUC_minus_TPR_gap = overall_AUC_minus_TPR_gap
        os.makedirs(f'{writer.log_dir}/{prefix}_max_overall_AUC_minus_TPR_gap', exist_ok=True)
        torch.save(model.state_dict(), f'{writer.log_dir}/{prefix}_max_overall_AUC_minus_TPR_gap/best_model.pth')
        print(
            f"Save model {prefix}_max_overall_AUC_minus_TPR_gap"
            f"{prefix}_max_overall_AUC_minus_TPR_gap= {overall_AUC_minus_TPR_gap :.4f})"
        )
        best_metrics[f'{prefix}_max_overall_AUC_minus_TPR_gap']  = max_overall_AUC_minus_TPR_gap

    del input_data, labels
    del val_df, val_df_logits, val_df_targets
    del val_metrics, val_loss
    del val_logits, val_preds, val_targets
    del largest_tpr_gap, worst_tpr
    del tprs, aucs
    torch.cuda.empty_cache()
    gc.collect()

    return best_metrics

def save_predictions(model, loader, filename_suffix, device, writer, metric_name):
    preds, targets, logits = test(model, loader, num_classes, device)
    df_preds = pd.DataFrame(data=preds, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df_preds, df_logits, df_targets], axis=1)
    df.to_csv(f'{writer.log_dir}/{metric_name}/{filename_suffix}.csv', index=False)
    del preds, targets, logits, df_preds, df_logits, df_targets, df
    torch.cuda.empty_cache()
    gc.collect()

def save_embeddings(model, loader, filename_suffix, device, writer, metric_name):
    embeds, targets = embeddings(model, loader, device)
    df_embeds = pd.DataFrame(data=embeds)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df_embeds, df_targets], axis=1)
    df.to_csv(f'{writer.log_dir}/{metric_name}/{filename_suffix}.csv', index=False)
    del embeds, targets, df_embeds, df_targets, df
    torch.cuda.empty_cache()
    gc.collect()