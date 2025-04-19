import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import joblib
from sklearn.metrics import accuracy_score, classification_report
import pytorch_lightning as pl
from simple_train import SignClassifier, SignMNISTDataset
from resnet_train import ResNetSignClassifier

def load_and_prepare_data():
    # Загружаем данные
    test_df = pd.read_csv("data/sign_mnist_test.csv")
    
    # Берем 5 жестов: A (0), B (1), L (11), W (22), Y (24)
    selected_labels = [0, 1, 11, 22, 24]
    label_map = {0: 0, 1: 1, 11: 2, 22: 3, 24: 4}
    
    # Выбираем нужные нам жесты из данных
    test_df = test_df[test_df['label'].isin(selected_labels)]
    
    # Подготовка данных для CNN и ResNet моделей
    X_test_cnn = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_test = test_df['label'].map(label_map).values
    
    # Подготовка данных для MediaPipe моделей
    X_test_mp = pd.read_csv("data/hand_keypoints_mediapipe_test.csv")
    if 'label' in X_test_mp.columns:
        y_test_mp = X_test_mp['label'].values
        X_test_mp = X_test_mp.drop("label", axis=1).values
    else:
        y_test_mp = y_test  # если метки сохранены отдельно
    
    # Трансформеры для изображений
    transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_resnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Создаем датасеты
    test_dataset_cnn = SignMNISTDataset(X_test_cnn, y_test, transform=transform_cnn)
    test_dataset_resnet = SignMNISTDataset(X_test_cnn, y_test, transform=transform_resnet)
    
    # Создаем DataLoader'ы
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=64)
    test_loader_resnet = DataLoader(test_dataset_resnet, batch_size=64)
    
    return {
        'X_test_mp': X_test_mp,
        'y_test_mp': y_test_mp,
        'test_loader_cnn': test_loader_cnn,
        'test_loader_resnet': test_loader_resnet,
        'y_test': y_test
    }

def evaluate_mediapipe_models(X_test, y_test):
    # Загружаем модели
    model_rf = joblib.load("models/rf_mediapipe.pkl")
    model_lgbm = joblib.load("models/lgbm_mediapipe.pkl")
    
    # Делаем предсказания
    y_pred_rf = model_rf.predict(X_test)
    y_pred_lgbm = model_lgbm.predict(X_test)
    
    # Вычисляем метрики
    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
    
    print("\nMediaPipe + RandomForest Results:")
    print(f"Accuracy: {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=['A', 'B', 'L', 'W', 'Y']))
    
    print("\nMediaPipe + LightGBM Results:")
    print(f"Accuracy: {acc_lgbm:.4f}")
    print(classification_report(y_test, y_pred_lgbm, target_names=['A', 'B', 'L', 'W', 'Y']))
    
    return {
        'rf_accuracy': acc_rf,
        'lgbm_accuracy': acc_lgbm
    }

def evaluate_cnn_model(test_loader, y_test):
    # Загружаем модель
    model = SignClassifier()
    model.load_state_dict(torch.load("models/sign_classifier_simple.pth"))
    model.eval()
    
    # Переносим модель на устройство (GPU если доступно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Делаем предсказания
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
    
    # Вычисляем метрики
    acc = accuracy_score(y_test, y_pred)
    
    print("\nSimple CNN Results:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['A', 'B', 'L', 'W', 'Y']))
    
    return {
        'cnn_accuracy': acc
    }

def evaluate_resnet_model(test_loader, y_test):
    # Загружаем модель
    model = ResNetSignClassifier()
    model.load_state_dict(torch.load("models/resnet_sign_classifier.pth"))
    model.eval()
    
    # Переносим модель на устройство (GPU если доступно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Делаем предсказания
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
    
    # Вычисляем метрики
    acc = accuracy_score(y_test, y_pred)
    
    print("\nResNet18 Results:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['A', 'B', 'L', 'W', 'Y']))
    
    return {
        'resnet_accuracy': acc
    }

def main():
    data = load_and_prepare_data()
    
    # Оцениваем модели
    print("Оцениваем модели на test...")
    print("===================================")
    
    # MediaPipe модели
    mp_results = evaluate_mediapipe_models(data['X_test_mp'], data['y_test_mp'])
    
    # Simple CNN
    cnn_results = evaluate_cnn_model(data['test_loader_cnn'], data['y_test'])
    
    # ResNet18
    resnet_results = evaluate_resnet_model(data['test_loader_resnet'], data['y_test'])
    
    print("\nSummary of all models:")
    print("=====================")
    print(f"MediaPipe + RandomForest Accuracy: {mp_results['rf_accuracy']:.4f}")
    print(f"MediaPipe + LightGBM Accuracy:    {mp_results['lgbm_accuracy']:.4f}")
    print(f"Simple CNN Accuracy:             {cnn_results['cnn_accuracy']:.4f}")
    print(f"ResNet18 Accuracy:               {resnet_results['resnet_accuracy']:.4f}")

if __name__ == "__main__":
    main()