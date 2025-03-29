import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

class QRCodeDataset(Dataset):
    def __init__(self, base_path='dataset', transform=None, augment=False):
        self.transform = transform
        self.augment = augment
        self.images = []
        self.labels = []
        
        # Load first prints (label 0)
        first_print_path = os.path.join(base_path, 'First Print')
        for img_name in os.listdir(first_print_path):
            if img_name.endswith('.png'):
                self.images.append(os.path.join(first_print_path, img_name))
                self.labels.append(0)
        
        # Load second prints (label 1)
        second_print_path = os.path.join(base_path, 'Second Print')
        for img_name in os.listdir(second_print_path):
            if img_name.endswith('.png'):
                self.images.append(os.path.join(second_print_path, img_name))
                self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def apply_augmentation(self, image):
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        contrast = np.random.uniform(0.8, 1.2)
        mean = np.mean(image)
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # Random Gaussian noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation during training
        if self.augment:
            image = self.apply_augmentation(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

class QRCodeCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(QRCodeCNN, self).__init__()
        
        # Convolutional layers with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.4)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        # Convolutional layers with residual connections
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        x5 = self.conv5(x4)
        
        # Global Average Pooling
        x = self.gap(x5)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        return x

class DeepLearningTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = QRCodeCNN().to(device)
        
        # Define transforms with normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                    verbose=True)
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Add L2 regularization
                l2_lambda = 0.001
                l2_reg = torch.tensor(0.).to(self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = train_loss / train_steps
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Accuracy: {val_acc:.4f}')
            print('------------------------')
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, val_accuracies)
    
    def evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        steps = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                steps += 1
        
        return total_loss / steps, correct / total
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create datasets with augmentation for training
    train_dataset = QRCodeDataset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ]), augment=True)
    
    # Validation dataset without augmentation
    val_dataset = QRCodeDataset(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ]), augment=False)
    
    # Split dataset
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize trainer
    trainer = DeepLearningTrainer()
    
    # Train model
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    print("Training complete!")

if __name__ == "__main__":
    main()