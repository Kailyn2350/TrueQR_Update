import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- 설정 (Configuration) ---
# [수정 필요 1] 이 값은 모델을 훈련할 때 사용했던 IMG_SIZE와 반드시 같아야 합니다.
# 예: 64x64로 훈련했다면 64, 96x96으로 훈련했다면 96으로 변경
IMG_SIZE = 64

# [수정 필요 2] 불러올 훈련된 모델 파일의 경로입니다.
# 이전 단계에서 저장한 모델 파일 이름으로 정확하게 지정해주세요.
MODEL_PATH = os.path.join("results_advanced", "qr_attention_model.keras")
NUM_EXAMPLES_TO_SHOW = 10


# --- [핵심 수정 부분] 이미지 전처리 함수 ---
# FFT 변환 부분을 제거하고, 훈련 시와 동일한 전처리(리사이즈, 정규화)만 수행합니다.
def preprocess_image_for_prediction(image_path, img_size):
    """예측을 위해 단일 이미지를 전처리합니다."""
    # 이미지를 흑백(grayscale)으로 불러옵니다.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 훈련 시와 동일하게 리사이즈합니다.
    img_resized = cv2.resize(img, (img_size, img_size))

    # 딥러닝 모델에 입력하기 위해 형태를 변환하고 정규화합니다.
    img_processed = img_resized / 255.0  # 픽셀 값 정규화 (0~1)
    img_processed = np.expand_dims(
        img_processed, axis=-1
    )  # 채널 차원 추가: (H, W) -> (H, W, 1)
    img_processed = np.expand_dims(
        img_processed, axis=0
    )  # 배치 차원 추가: (H, W, 1) -> (1, H, W, 1)

    return img_processed


# --- 메인 예측 로직 (수정 없음) ---
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 훈련된 모델을 찾을 수 없습니다. 경로: {MODEL_PATH}")
        return

    model = load_model(MODEL_PATH)
    print(f"모델을 '{MODEL_PATH}'에서 불러왔습니다.")

    # 테스트할 이미지가 들어있는 폴더 경로 (실제 경로로 수정 필요)
    true_dir = "Data/true"  # 테스트용 정품 사진 폴더
    false_dir = "Data/false"  # 테스트용 위조 사진 폴더

    if not os.path.exists(true_dir) or not os.path.exists(false_dir):
        print(
            f"오류: 테스트 데이터 폴더를 찾을 수 없습니다. '{true_dir}' 와 '{false_dir}' 경로를 확인해주세요."
        )
        return

    true_images = [os.path.join(true_dir, f) for f in os.listdir(true_dir)]
    false_images = [os.path.join(false_dir, f) for f in os.listdir(false_dir)]

    if not true_images or not false_images:
        print("오류: 테스트 폴더에 이미지가 없습니다.")
        return

    # 테스트할 이미지들을 랜덤하게 선택
    test_images = random.sample(
        true_images, k=NUM_EXAMPLES_TO_SHOW // 2
    ) + random.sample(false_images, k=NUM_EXAMPLES_TO_SHOW // 2)
    random.shuffle(test_images)

    plt.figure(figsize=(20, 10))

    for i, image_path in enumerate(test_images):
        # Ground Truth 라벨 설정
        ground_truth = "True" if os.path.dirname(image_path) == true_dir else "False"

        # 수정한 전처리 함수를 사용하여 이미지 처리
        processed_img = preprocess_image_for_prediction(image_path, IMG_SIZE)

        if processed_img is not None:
            # 모델로 예측 수행
            prediction_prob = model.predict(processed_img)[0][0]
            prediction = "True" if prediction_prob > 0.5 else "False"

            # 결과 시각화
            original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            plt.subplot(2, 5, i + 1)
            plt.imshow(original_img, cmap="gray")

            # 예측이 맞았는지 틀렸는지에 따라 제목 색상 변경
            title_color = "green" if prediction == ground_truth else "red"
            plt.title(
                f"GT: {ground_truth}\nPred: {prediction} ({prediction_prob:.2f})",
                color=title_color,
            )

            plt.xticks([]), plt.yticks([])
        else:
            print(f"이미지를 처리할 수 없습니다: {image_path}")

    plt.tight_layout()
    result_path = "results_advanced/prediction_results.png"
    plt.savefig(result_path)
    print(f"예측 결과 이미지가 '{result_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
