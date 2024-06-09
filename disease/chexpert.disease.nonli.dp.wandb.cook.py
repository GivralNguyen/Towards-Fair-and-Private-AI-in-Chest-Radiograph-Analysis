
import torch
import os
from skimage.io import imsave
from dataset import CheXpertDataModule
from model import NonLiResNetDP
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
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
import os
from utils import num_classes, process_validation, save_predictions, save_embeddings
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) #shut tqdm up

image_size = (224, 224)
torch.set_float32_matmul_precision('high')
DELTA = 1/76205
batch_size = 4096 #
# max_physical_batch_size = 32
num_workers = 4
MAX_GRAD_NORM = 1.2


def train_model(model, train_loader, val_loader, val_resample_loader, val_resample_sex_loader, optimizer, writer, epoch, max_physical_batch_size, privacy_engine, best_metrics):

    model.train()
    """
    Traning part of the code
    """
    losses = []
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=max_physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
        train_bar = tqdm(memory_safe_data_loader, desc='Training', unit='batch')
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
    epsilon = privacy_engine.get_epsilon(DELTA)

    # Print the training metrics
    print(f"\tTrain Epoch: {epoch} \tTrain Loss: {np.mean(losses):.6f} (ε = {epsilon:.2f}, δ = {DELTA})", end=' ')
    # Log metrics to wandb
    wandb.log({
        "train_loss": np.mean(losses),
        "Epsilon": epsilon,
        "epoch": epoch,
    })
    
    """
    Evaluation part of the code
    """
    model.eval()
    best_metrics = process_validation(val_loader, model , '../data/chexpert.sample.val.csv', writer, "val", epoch , best_metrics)
    best_metrics = process_validation(val_resample_loader, model, '/vol/aimspace/users/ngq/Towards-Fair-and-Private-AI-in-Chest-Radiograph-Analysis/data/chexpert.resample.val.csv', writer, "val_resample", epoch , best_metrics)
    best_metrics = process_validation(val_resample_sex_loader, model, '/vol/aimspace/users/ngq/Towards-Fair-and-Private-AI-in-Chest-Radiograph-Analysis/data/chexpert.resample.sex.val.csv', writer, "val_resample_sex", epoch, best_metrics)
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
        EPSILON = config.EPSILON
        # datacsv_test_sex_img_resample
        data = CheXpertDataModule(csv_train_img='../data/chexpert.sample.train.csv',
                                csv_val_img='../data/chexpert.sample.val.csv',
                                csv_val_img_resample='../data/chexpert.resample.val.csv',
                                csv_val_sex_img_resample='../data/chexpert.resample.sex.val.csv',
                                csv_test_img='../data/chexpert.sample.test.csv',
                                csv_test_img_resample='../data/chexpert.resample.test.csv',
                                csv_test_sex_img_resample='../data/chexpert.resample.sex.test.csv',
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
        # print(model)
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


        best_metrics = {}
        prefixes = ['val','val_resample', 'val_resample_sex']

        for prefix in prefixes:
            best_metrics[f'{prefix}_max_overall_AUC'] = 0.0
            best_metrics[f'{prefix}_max_overall_AUC_minus_TPR_gap'] = 0.0
            # best_metrics[f'{prefix}_min_largest_TPR_gap'] = float('inf')
            best_metrics[f'{prefix}_max_worse_case_group_TPR'] = 0.0

        for epoch in tqdm(range(epochs), desc="Epoch", unit="epoch"):
            best_metrics = train_model(model, train_loader,data.val_dataloader() ,data.val_resample_dataloader(), data.val_sex_resample_dataloader(),optimizer=optimizer, writer=writer, epoch = epoch + 1, max_physical_batch_size = config.max_physical_batch_size, privacy_engine=privacy_engine, best_metrics = best_metrics)
        # train_model(model,train_loader,data.val_dataloader() ,optimizer=optimizer, writer=writer)

        for metric_name, metric_value in best_metrics.items():
            loaded_state_dict = torch.load(f'{writer.log_dir}/{metric_name}/best_model.pth')
            model.load_state_dict(loaded_state_dict)
            print(f'{writer.log_dir}/{metric_name}/best_model.pth')

            print('VALIDATION')
            save_predictions(model, data.val_dataloader(), 'predictions.val', device, writer, metric_name)

            print('VALIDATION RESAMPLE')
            save_predictions(model, data.val_resample_dataloader(), 'predictions.resample.val', device, writer, metric_name)

            print('VALIDATION RESAMPLE SEX')
            save_predictions(model, data.val_sex_resample_dataloader(), 'predictions.resample.sex.val', device, writer, metric_name)

            print('TESTING')
            save_predictions(model, data.test_dataloader(), 'predictions.test', device, writer, metric_name)

            print('TESTING RESAMPLE')
            save_predictions(model, data.test_resample_dataloader(), 'predictions.resample.test', device, writer, metric_name)

            print('TESTING RESAMPLE SEX')
            save_predictions(model, data.test_sex_resample_dataloader(), 'predictions.resample.sex.test', device, writer, metric_name)

            print('EMBEDDINGS VAL')
            head = model._module.remove_head()
            save_embeddings(model, data.val_dataloader(), 'embeddings.val', device, writer, metric_name)

            print('EMBEDDINGS VAL RESAMPLE')
            save_embeddings(model, data.val_resample_dataloader(), 'embeddings.resample.val', device, writer, metric_name)

            print('EMBEDDINGS VAL SEX RESAMPLE')
            save_embeddings(model, data.val_sex_resample_dataloader(), 'embeddings.resample.sex.val', device, writer, metric_name)

            print('EMBEDDINGS TEST')
            save_embeddings(model, data.test_dataloader(), 'embeddings.test', device, writer, metric_name)

            print('EMBEDDINGS TEST RESAMPLE')
            save_embeddings(model, data.test_resample_dataloader(), 'embeddings.resample.test', device, writer, metric_name)

            print('EMBEDDINGS TEST SEX RESAMPLE')
            save_embeddings(model, data.test_sex_resample_dataloader(), 'embeddings.resample.sex.test', device, writer, metric_name)

            model._module.restore_head(head)
                        

            # del new_state_dict, new_state_dict_without_fc, pretrained_resnet18
            # torch.cuda.empty_cache()
            # gc.collect()
            # print("after releasing memory")
            # print(torch.cuda.memory_summary())
        del model, optimizer, train_loader, data, privacy_engine, loaded_state_dict
        del new_state_dict, new_state_dict_without_fc, pretrained_resnet18, head
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