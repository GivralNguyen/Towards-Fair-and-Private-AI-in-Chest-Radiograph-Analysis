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
from skimage.io import imread, imsave
from tqdm import tqdm
from argparse import ArgumentParser
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.lightning import DPLightningDataModule
from opacus.utils.batch_memory_manager import BatchMemoryManager
from dataset import CheXpertDataModule, CheXpertDataset
from groupnormresnet import resnet18gn

# Set up data information
image_size = (224, 224)
num_classes = 14
epochs = 20
num_workers = 4
img_data_dir = '/vol/aimspace/projects/CheXpert/CheXpert/'
torch.set_float32_matmul_precision('high')

