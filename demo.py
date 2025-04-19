import cv2
import numpy as np
import torch
import joblib
from torchvision import transforms
import mediapipe as mp
import argparse

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Словарь для соответствия меток и жестов
GESTURE_NAMES = {
    0: "A",
    1: "B",
    2: "L",
    3: "W",
    4: "Y"
}

def load_model(model_type):
    """Загрузка выбранной модели"""
    if model_type == 'rf_mediapipe':
        model = joblib.load('models/rf_mediapipe.pkl')
    elif model_type == 'lgbm_mediapipe':
        model = joblib.load('models/lgbm_mediapipe.pkl')
    elif model_type == 'simple_cnn':
        from simple_train import SignClassifier
        model = SignClassifier()
        model.load_state_dict(torch.load('models/sign_classifier_simple.pth', map_location=torch.device('cpu')))
        model.eval()
    elif model_type == 'resnet':
        from resnet_train import ResNetSignClassifier
        model = ResNetSignClassifier()
        model.load_state_dict(torch.load('models/resnet_sign_classifier.pth', map_location=torch.device('cpu')))
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def extract_mediapipe_features(image):
    """Извлечение ключевых точек с помощью MediaPipe"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    hand_landmarks = results.multi_hand_landmarks[0]
    keypoints = []
    for landmark in hand_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(keypoints).reshape(1, -1)

def preprocess_image_for_cnn(image, model_type):
    """Подготовка изображения для CNN моделей"""
    # Конвертируем в grayscale и ресайзим
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255.0
    
    if model_type == 'simple_cnn':
        # Для простой CNN: [1, 1, 28, 28]
        image = np.expand_dims(image, axis=0)  # Добавляем размерность канала
        image = torch.from_numpy(image).unsqueeze(0)
        transform = transforms.Normalize((0.5,), (0.5,))
        image = transform(image)
    else:  # Для ResNet
        # Для ResNet: [1, 3, 28, 28]
        image = np.stack((image,)*3, axis=0)  # Делаем 3 канала
        image = torch.from_numpy(image)
        transform = transforms.Normalize((0.5,), (0.5,))
        image = transform(image).unsqueeze(0)
    
    return image

def predict_gesture(model, model_type, frame):
    """Предсказание жеста на основе типа модели"""
    if 'mediapipe' in model_type:
        features = extract_mediapipe_features(frame)
        if features is not None:
            pred = model.predict(features)[0]
            return GESTURE_NAMES[pred]
        return None
    else:
        input_tensor = preprocess_image_for_cnn(frame, model_type)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
            return GESTURE_NAMES[pred]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['rf_mediapipe', 'lgbm_mediapipe', 'simple_cnn', 'resnet'],
                       help='Model type to use for prediction')
    args = parser.parse_args()

    # Загрузка модели
    model = load_model(args.model)
    print(f"Loaded {args.model} model")

    # Инициализация веб-камеры
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Предсказание жеста
        gesture = predict_gesture(model, args.model, frame)
        
        # Отображение результата
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Sign Language Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()