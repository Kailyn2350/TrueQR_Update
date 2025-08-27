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
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    Reshape,
    multiply,
    Conv2D,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers

# --- 설정 (Configuration) ---
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
RESULTS_DIR = "results_224_diverse_data_update_aug"


# --- 데이터 로딩 함수 (이전과 동일) ---
def load_data(true_dir, false_dir, img_size):
    images, labels = [], []  # 빈 리스트 초기화
    # ... (이하 동일)
    for filename in os.listdir(
        true_dir
    ):  # os.listdir(주소) 주소 폴더에 있는 파일 이름을 리스트에 저장 그후 반복문으로 파일 이름을 하나씩 불러옴
        img_path = os.path.join(
            true_dir, filename
        )  # 저번에 배웠던 os.path.join은 두가지 합쳐서 경로를 만듦
        img = cv2.imread(
            img_path, cv2.IMREAD_GRAYSCALE
        )  # cv2에 있는 imread(image read)를 이용해서 이미지를 읽어옴 cv2.IMREAD_GRAYSCALE은 흑백으로 읽어옴 2차원 배열
        if img is not None:  # 모든 이미지를 224x224사이즈로 통일하기 위해서
            img = cv2.resize(
                img, (img_size, img_size)
            )  # 이미지 한장을 224x224로 크기를 조정
            images.append(img)  # 리사이즈한 이미지를 초기화했던 리스트에 추가
            labels.append(1)  # 정답이니까 1라벨을 추가
    for filename in os.listdir(false_dir):  # 위랑 동일하게 이번에는 위조 이미지
        img_path = os.path.join(false_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(0)
    images = np.array(images).reshape(-1, img_size, img_size, 1) / 255.0
    # np.array(images)는 기존에 images는 python리스트인데 이거를 딥러닝 모델에 넣을려면 numpy 배열로 변환해야함
    # -1은 N(전체 이미지 개수)는 자동으로 계산하고, img_size, img_size는 이미지의 크기, 1은 흑백 이미지이므로 채널 수를 1로 설정
    # 255로 나누어서 픽셀 값을 0과 1 사이로 정규화
    labels = np.array(labels)
    # 동일하게 labels를 numpy 배열로 변환
    return images, labels


# --- 어텐션(SE Block) 함수 (이전과 동일) ---
# 아래에 있는 어텐션 함수는 어떤 모델이 들어오면 그것에 어텐션을 적용하는 함수임 이번의 경우에는 MobileNetV2에 적용하는데
# 원래의 SE-Net방식을 이용하면 모든 층의 출력값에 어텐션을 적용해야됨, 하지만 이번에는 간단하게 최종 출력층에만 어텐션을 적용함
def squeeze_excite_block(input_tensor):
    # ... (이하 동일)
    channels = input_tensor.shape[
        -1
    ]  # CNN의 최종 출력층이 어텐션의 입력 특징맵이 됨 그 입력 특징맵의 마지막 요소인 채널 수를 가져옴 왜냐하면 SE Block은 채널 어텐션 모델이기 때문임
    # 여기서 채널이란 특징을 말하는데 흑백은 (Gray), 컬러 이미지는 (RGB)로 표현됨 그러니까 이미지를 설명하는 특징의 요소가 몇개인지 알 수 있음
    # CNN을 거치면서 이런 채널을 여러개 생성하게 됨 예를들어서 어떤 채널은 수직선을 감지하고 어떤 채널은 둥근 모양을 감지하는 등
    se = GlobalAveragePooling2D()(input_tensor)
    # CNN에서 나온 특징맵 input_tensor가 (batch, H, W, C)이렇게 생겼는데 거기서 GlobalAveragePooling2D를 하는걸로 H x W를 채널별로 평균을 내서 (batch, C)로 만들어줌
    se = Reshape((1, 1, channels))(se)
    # input_tensor로써 (B, H, W, C)이것을 받았기 때문에 위에서 (batch, c)로 만들었던거를 다시 (B, 1, 1, C)로 만들어줌. B, 1, 1, channels라고 적지 않고 1, 1, channels라고 적는 이유는 B는 배치 차원인데 여러 모델에 사용가능하게 하기 위해서 생략하는걸로 자동으로 유지되게 함
    se = Dense(
        channels // 8, activation="relu", kernel_initializer="he_normal", use_bias=False
    )(se)
    # Dense의 첫번째 요소는 출력 뉴런의 개수인데 그냥 1280을 그대로 사용하게 된다면 입력 1280 출력 1280으로 1280x1280의 연산이 필요하게 됨. 그래서 연산을 줄이기 위해서 채널을 8로 나눠서 압축 시킴
    # kernel_initializer는 가중치 초기화를 위한 것으로, "he_normal"은 Relu 활성화 함수에 적합한 초기화 방법임. 방법으로는 평균이 0이고 표준편차가 sqrt(2/n)인 정규분포를 따르는 가중치로 초기화함
    se = Dense(
        channels, activation="sigmoid", kernel_initializer="he_normal", use_bias=False
    )(se)
    # 전의 Dense에서 채널을 8로 나눴으니까 이번에는 다시 원래대로 돌려놓기 위해서 채널을 8배로 늘림. 활성화 함수는 시그모이드로 0과 1사이의 값을 출력함
    return multiply([input_tensor, se])


def main():
    # ... (데이터 로딩 및 모델 구성 부분은 이전과 동일)
    train_true = "Data/True_aug"
    train_false = "Data/False_aug"
    val_true = "Data/True/val"  # 검증셋은 보통 증강하지 않음
    val_false = "Data/False/val"

    X_train, y_train = load_data(train_true, train_false, IMG_SIZE)
    X_val, y_val = load_data(val_true, val_false, IMG_SIZE)
    inputs = Input(
        shape=(IMG_SIZE, IMG_SIZE, 1)
    )  # keras에서 입력 레이어를 정의하는 부분
    x = Conv2D(3, (3, 3), padding="same", activation="relu")(
        inputs
    )  # MobileNetV2라는 사전학습된 모델을 이용하는데 해당 모델은 3채널 컬러 이미지를 입력으로 받기 때문에 Conv2D를 이용해서 1채널 흑백 이미지를 3채널로 변환함
    # 앞에 있는 3, (3, 3)은 3채널로 3x3커널로 기존에 있는 입력을 3x3 커널로 특정 패턴을 감지해서 3채널로 변환. 근데 224x224에 3x3의 커널을 적용하면 222x222가 되는데 padding="same"을 사용해서 입력과 출력의 크기를 동일하게 유지함
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        # include_top : MobileNetV2의 맨 위 (Top) Fully Connected 층을 포함할지 말지 결정
        # Convolution은 특정 영역만 보고 계산 but Fully Connected는 입력 전체(Flatten된 특징맵 전부)를 연결해서 출력
        # imagenet : ImageNet 데이터셋으로 사전 학습된 가중치(Weights)를 불러오겠다
    )
    base_model.trainable = False  # 이 모델의 가중치를 학습 중에 업데이트하지 않겠다 -> 특징 추출하는데만 쓰고 학습은 우리가 새로 붙이는 층에서만
    x = base_model(
        x, training=False
    )  # MobileNetV2 내부의 BatchNorm, Dropout 같은 층들이 “학습 모드” 대신 “추론 모드”로 동작하게 설정
    x = squeeze_excite_block(
        x
    )  # 어텐션 적용 MobileNetV2에서 나온 추론값(고차원 특징맵)을 어텐션에 적용
    x = GlobalAveragePooling2D()(
        x
    )  # MobileNetV2가 뽑은 “공간적 특징맵”을 “벡터” 형태로 만들어서 FC(Dense) 층에 넣을 수 있게 함 (B, H, W, C) -> (B, C)
    x = Dropout(0.3)(x)  # 30% 정도 랜덤으로 뉴런을 꺼서 과적합 방지
    outputs = Dense(1, activation="sigmoid")(
        x
    )  # Fully Connected 구조의 마지막 층으로, 1개의 뉴런을 사용하여 이진 분류를 수행함. 활성화 함수는 시그모이드로 0과 1 사이의 값을 출력함
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),  # Adam 옵티마이저(최적화 알고리즘)를 사용하고, 학습률은 0.001로 설정
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),  # 검증 데이터 손실이 10번 변하지 않으면 학습을 중단하고, 가장 좋은 가중치를 복원
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
        ),  # 검증 데이터 손실이 개선되지 않으면 학습률을 20% 줄임
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
    plt.figure(figsize=(12, 5))  # 그래프 크기 설정
    plt.subplot(
        1, 2, 1
    )  # 위에서 정한 figure에 2개의 그래프를 넣기 위해서 1행 2열로 나누고 첫번째 그래프를 그릴 위치 설정
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
    # classification_report가 출력하는거는
    # 1. 정밀도(Precision): 모델이 True로 예측한 것 중 실제 True인 비율
    # 2. 재현율(Recall): 실제 True인 것 중 모델이 True로 예측한 비율
    # 3. F1 점수: 정밀도와 재현율의 조화 평균
    # 4. 지원 수(Support): 각 클래스의 실제 샘플 수
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
    # 혼동 행렬(Confusion Matrix) = 실제 라벨 vs 예측 라벨을 표로 정리해서, 모델이 맞춘/틀린 분류 결과를 한눈에 보여주는 도구


if __name__ == "__main__":
    main()
