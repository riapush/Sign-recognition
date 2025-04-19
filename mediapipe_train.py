import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Define the label mapping
GESTURE_NAMES = {
    0: "A",
    1: "B",
    2: "L",
    3: "W",
    4: "Y"
}

# Map original labels to new labels (0-4)
LABEL_MAP = {
    0: 0,   # A
    1: 1,   # B
    11: 2,  # L
    22: 3,  # W
    24: 4   # Y
}

def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Extract all 21 hand landmarks (x,y,z coordinates)
    hand_landmarks = results.multi_hand_landmarks[0]
    keypoints = []
    for landmark in hand_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(keypoints)

def process_dataframe(df, dataset_name):
    processed_df = pd.DataFrame()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
        # Convert MNIST image to 3-channel BGR
        image = (row.values[1:].reshape(28, 28) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Resize to a larger size for better MediaPipe detection
        image = cv2.resize(image, (224, 224))
        
        temp_path = f"temp_{dataset_name}_{idx}.jpg"
        cv2.imwrite(temp_path, image)
        
        # Extract keypoints
        keypoints = extract_keypoints(temp_path)
        os.remove(temp_path)
        
        if keypoints is not None:
            # Map the original label to new label (0-4)
            mapped_label = LABEL_MAP[row["label"]]
            data = {"label": mapped_label}
            for i, coord in enumerate(keypoints):
                data[f"kp_{i}"] = coord
            processed_df = pd.concat([processed_df, pd.DataFrame([data])], ignore_index=True)
    
    return processed_df

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv("data/sign_mnist_train.csv")
    test_df = pd.read_csv("data/sign_mnist_test.csv")

    # Select 5 gestures: A (0), B (1), L (11), W (22), Y (24)
    selected_labels = [0, 1, 11, 22, 24]

    # Filter selected gestures
    train_df = train_df[train_df['label'].isin(selected_labels)]
    test_df = test_df[test_df['label'].isin(selected_labels)]

    # Process both datasets
    train_processed = process_dataframe(train_df, "train")
    test_processed = process_dataframe(test_df, "test")

    # Save keypoints for reuse
    train_processed.to_csv("data/hand_keypoints_mediapipe_train.csv", index=False)
    test_processed.to_csv("data/hand_keypoints_mediapipe_test.csv", index=False)

    # Prepare data for training
    X_train = train_processed.drop("label", axis=1).values
    y_train = train_processed["label"].values

    X_test = test_processed.drop("label", axis=1).values
    y_test = test_processed["label"].values

    # Train RandomForest
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X_train, y_train)

    # Train LightGBM
    model_lgbm = LGBMClassifier()
    model_lgbm.fit(X_train, y_train)

    # Evaluate models
    y_pred_rf = model_rf.predict(X_test)
    y_pred_lgbm = model_lgbm.predict(X_test)

    print(f"RandomForest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
    print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgbm):.2f}")

    # Save models and label mapping
    joblib.dump(model_lgbm, "models/lgbm_mediapipe.pkl")
    joblib.dump(model_rf, "models/rf_mediapipe.pkl")
    joblib.dump(LABEL_MAP, "models/label_map.pkl")  # Save the label mapping for later use

    # Release MediaPipe resources
    hands.close()