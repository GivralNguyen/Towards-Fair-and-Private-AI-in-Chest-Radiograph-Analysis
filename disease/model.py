
from groupnormresnet import resnet18gn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18

from tqdm import tqdm
from opacus.utils.batch_memory_manager import BatchMemoryManager
# Instantiate the custom model

class NonLiResNet(nn.Module):
    def __init__(self, num_classes, epochs, writer):
        super().__init__()
        self.num_classes = num_classes
        self.model = resnet18(weights='ResNet18_Weights.DEFAULT')
        self.epochs = epochs
        self.writer = writer
        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)


    def remove_head(self):
        num_features = self.model.fc.in_features
        id_layer = nn.Identity(num_features)
        self.model.fc = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.0001, weight_decay=1e-5)
        return optimizer

    def train_model(self, train_loader, val_loader, optimizer):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            best_val_loss = float('inf')  # Initialize with a large value
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='batch')
            for batch in train_bar:
                optimizer.zero_grad()
                input_data, labels = batch['image'].to('cuda'), batch['label'].to('cuda')
                output = self.forward(input_data)
                prob = torch.sigmoid(output)
                loss = F.binary_cross_entropy(prob, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix({'Train Loss': train_loss / len(train_loader)})

            train_loss /= len(train_loader)
            self.writer.add_scalar('Train/Loss', train_loss, epoch)

            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_data, labels = batch['image'].to('cuda'), batch['label'].to('cuda')
                    output = self.forward(input_data)
                    prob = torch.sigmoid(output)
                    val_loss += F.binary_cross_entropy(prob, labels)
            val_loss /= len(val_loader)
            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.writer.log_dir}/best_model.pth')
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)

class NonLiResNetDP(NonLiResNet):
    def __init__(self,
        num_classes, 
        epochs,
        writer,
        ):
        super().__init__(
        num_classes = num_classes,
        epochs = epochs,
        writer = writer,
        )
        self.model = resnet18gn()
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

