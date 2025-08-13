import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    Reshape,
    multiply,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 설정 (Configuration) ---
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 50
RESULTS_DIR = "results_advanced"


# --- 데이터 로딩 함수 (이전과 동일) ---
def load_data(true_dir, false_dir, img_size):
    images, labels = [], []
    # ... (이하 동일)
    for filename in os.listdir(true_dir):
        img_path = os.path.join(true_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(1)
    for filename in os.listdir(false_dir):
        img_path = os.path.join(false_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(0)
    images = np.array(images).reshape(-1, img_size, img_size, 1) / 255.0
    labels = np.array(labels)
    return images, labels


# --- 어텐션(SE Block) 함수 (이전과 동일) ---
def squeeze_excite_block(input_tensor):
    # ... (이하 동일)
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(
        channels // 8, activation="relu", kernel_initializer="he_normal", use_bias=False
    )(se)
    se = Dense(
        channels, activation="sigmoid", kernel_initializer="he_normal", use_bias=False
    )(se)
    return multiply([input_tensor, se])


def main():
    # ... (데이터 로딩 및 모델 구성 부분은 이전과 동일)
    true_dir = "Data/train_data_true"
    false_dir = "Data/train_data_false"
    if not os.path.exists(true_dir) or not os.path.exists(false_dir):
        print(f"오류: 데이터 폴더를 찾을 수 없습니다.")
        return
    images, labels = load_data(true_dir, false_dir, IMG_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(3, (3, 3), padding="same", activation="relu")(inputs)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = squeeze_excite_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001),
    ]
    print("고급 모델 훈련을 시작합니다...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # --- [추가된 부분] 훈련 결과 시각화 및 상세 리포트 저장 ---
    print("\n--- 최종 결과 평가 및 저장 시작 ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. 모델 저장 (.keras 형식 권장)
    model_save_path = os.path.join(RESULTS_DIR, "qr_attention_model.keras")
    model.save(model_save_path)
    print(f"모델이 '{model_save_path}'에 저장되었습니다.")

    # 2. 훈련 과정(Accuracy, Loss) 그래프 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    history_plot_path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.tight_layout()
    plt.savefig(history_plot_path)
    print(f"훈련 과정 그래프가 '{history_plot_path}'에 저장되었습니다.")
    plt.close()  # 그래프 창 닫기

    # 3. 검증 데이터에 대한 예측 수행
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # 4. 분류 리포트(Classification Report) 저장
    report = classification_report(
        y_val, y_pred, target_names=["False (위조)", "True (정품)"]
    )
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("--- 분류 리포트 ---\n\n")
        f.write(report)
    print(f"분류 리포트가 '{report_path}'에 저장되었습니다.")
    print(report)

    # 5. 혼동 행렬(Confusion Matrix) 시각화 및 저장
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["False (위조)", "True (정품)"],
        yticklabels=["False (위조)", "True (정품)"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"혼동 행렬 이미지가 '{cm_path}'에 저장되었습니다.")
    plt.close()


if __name__ == "__main__":
    main()
