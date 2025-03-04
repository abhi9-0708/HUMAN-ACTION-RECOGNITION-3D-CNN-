import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv3D, MaxPooling3D, BatchNormalization, 
                                    Dense, Dropout, Flatten, Input, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Parameters
DATASET_PATH = '/fab3/btech/2022/abhinav.kumar22b/miniconda/kth'
NUM_FRAMES = 16
FRAME_SIZE = (16, 16)  # Updated input size
CHANNELS = 1  # Grayscale frames for spatial stream
EPOCHS = 50
BATCH_SIZE = 16
K_FOLDS = 5

# Function to extract frames (for spatial stream)
def extract_frames(video_path, num_frames, frame_size):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        return None

    indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        return None

    frames = np.array(frames, dtype=np.float32) / 255.0
    frames = np.expand_dims(frames, axis=-1)  # Add channel dimension
    if frames.shape != (num_frames, frame_size[0], frame_size[1], 1):
        return None

    return frames

# Function to extract optical flow (for temporal stream)
def extract_optical_flow(video_path, num_frames, frame_size):
    cap = cv2.VideoCapture(video_path)
    flows = []

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames + 1:
        cap.release()
        return None

    indices = np.linspace(0, total_frames-2, num=num_frames, dtype=int)  # Optical flow needs two frames

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame1 = cap.read()
        if not ret:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx+1)
        ret, frame2 = cap.read()
        if not ret:
            continue

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.resize(flow, (frame_size[1], frame_size[0]))

        # Convert 2-channel optical flow to 3-channel by padding
        flow = np.pad(flow, ((0, 0), (0, 0), (0, 1)), mode='constant')  
        flows.append(flow)

    cap.release()

    if len(flows) < num_frames:
        return None

    flows = np.array(flows, dtype=np.float32)
    if flows.shape != (num_frames, frame_size[0], frame_size[1], 3):
        return None

    return flows

# Load dataset for both spatial and temporal streams
def load_dataset(dataset_path, num_frames, frame_size):
    classes = sorted(os.listdir(dataset_path))
    X_spatial, X_temporal, y = [], [], []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for video_file in [f for f in os.listdir(class_dir) if f.endswith(('.avi', '.mp4'))]:
            video_path = os.path.join(class_dir, video_file)

            frames = extract_frames(video_path, num_frames, frame_size)
            flows = extract_optical_flow(video_path, num_frames, frame_size)

            if frames is not None and flows is not None:
                X_spatial.append(frames)
                X_temporal.append(flows)
                y.append(class_idx)

    X_spatial = np.array(X_spatial, dtype=np.float32)
    X_temporal = np.array(X_temporal, dtype=np.float32)
    y = to_categorical(np.array(y), num_classes=len(classes))

    return X_spatial, X_temporal, y, classes

# Define the Two-Stream CNN Model
def create_two_stream_model(input_shape_spatial, input_shape_temporal, num_classes):
    # Spatial Stream (Grayscale Frames)
    spatial_input = Input(shape=input_shape_spatial, name='spatial_input')
    x = Conv3D(32, (5, 5, 5), activation='relu', padding='same')(spatial_input)
    x = Conv3D(32, (5, 5, 5), activation='relu', padding='same')(x)
    x = MaxPooling3D((3, 3, 3))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    spatial_output = Dense(256, activation='relu')(x)

    # Temporal Stream (Optical Flow)
    temporal_input = Input(shape=input_shape_temporal, name='temporal_input')
    y = Conv3D(32, (5, 5, 5), activation='relu', padding='same')(temporal_input)
    y = Conv3D(32, (5, 5, 5), activation='relu', padding='same')(y)
    y = MaxPooling3D((3, 3, 3))(y)
    y = Dropout(0.5)(y)
    y = Flatten()(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(0.5)(y)
    temporal_output = Dense(256, activation='relu')(y)

    # Concatenate both streams
    merged = Concatenate()([spatial_output, temporal_output])
    merged = Dense(1024, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(num_classes, activation='softmax')(merged)

    # Create the model
    model = Model(inputs=[spatial_input, temporal_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Load dataset
X_spatial, X_temporal, y, class_names = load_dataset(DATASET_PATH, NUM_FRAMES, FRAME_SIZE)
print(f"Loaded dataset with shape: {X_spatial.shape}, {X_temporal.shape}, Classes: {len(class_names)}")

# K-Fold Cross Validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

fold_no = 1
all_accuracies = []
all_precision = []
all_recall = []
all_f1 = []

for train_idx, val_idx in kf.split(X_spatial):
    print(f"\nTraining on Fold {fold_no}/{K_FOLDS}")

    X_spatial_train, X_spatial_val = X_spatial[train_idx], X_spatial[val_idx]
    X_temporal_train, X_temporal_val = X_temporal[train_idx], X_temporal[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_two_stream_model(
        input_shape_spatial=(NUM_FRAMES, FRAME_SIZE[0], FRAME_SIZE[1], 1),
        input_shape_temporal=(NUM_FRAMES, FRAME_SIZE[0], FRAME_SIZE[1], 3),
        num_classes=len(class_names)
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train
    model.fit([X_spatial_train, X_temporal_train], y_train, 
              validation_data=([X_spatial_val, X_temporal_val], y_val), 
              epochs=EPOCHS, batch_size=BATCH_SIZE, 
              callbacks=[early_stopping], verbose=1)

    # Evaluate
    y_pred = model.predict([X_spatial_val, X_temporal_val])
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_val, axis=1)

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    class_report = classification_report(y_true_labels, y_pred_labels, output_dict=True)

    precision = np.mean([class_report[str(i)]['precision'] for i in range(len(class_names))])
    recall = np.mean([class_report[str(i)]['recall'] for i in range(len(class_names))])
    f1 = np.mean([class_report[str(i)]['f1-score'] for i in range(len(class_names))])

    all_accuracies.append(accuracy)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)

    print(f"\n--- Fold {fold_no} Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels))

    fold_no += 1

# Average Performance
print("\n=== Final Performance (Averaged Over All Folds) ===")
print(f"Average Accuracy: {np.mean(all_accuracies):.4f}")
print(f"Average Precision: {np.mean(all_precision):.4f}")
print(f"Average Recall: {np.mean(all_recall):.4f}")
print(f"Average F1-Score: {np.mean(all_f1):.4f}")