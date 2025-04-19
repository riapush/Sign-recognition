import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torch.nn.functional as F

class SignMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class SignClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == "__main__":
    # Загружаем данные
    train_df = pd.read_csv("data/sign_mnist_train.csv")
    test_df = pd.read_csv("data/sign_mnist_test.csv")

    # Берем 5 жестов: A (0), B (1), L (11), W (22), Y (24)
    selected_labels = [0, 1, 11, 22, 24]
    label_map = {0: 0, 1: 1, 11: 2, 22: 3, 24: 4}

    # Выбираем нужные нам жесты из данных
    train_df = train_df[train_df['label'].isin(selected_labels)]
    test_df = test_df[test_df['label'].isin(selected_labels)]

    X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = train_df['label'].map(label_map).values
    X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test = test_df['label'].map(label_map).values

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = SignMNISTDataset(X_train, y_train, transform=transform)
    test_dataset = SignMNISTDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Обучаем модель и сохраняем ее
    model = SignClassifier()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='./',
        filename='models/best_sign_classifier',
        save_top_k=1,
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        accelerator='auto',
        deterministic=True
    )
    trainer.fit(model, train_loader, test_loader)

    torch.save(model.state_dict(), 'models/sign_classifier_simple.pth')