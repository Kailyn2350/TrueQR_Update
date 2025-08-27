import os
import random
import cv2
import numpy as np
import hashlib
from pathlib import Path

# 해시를 계산해주는 표준 라이브러리
from typing import List, Tuple

# ------------ 설정 ----------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

AUG_CONFIG = {
    "p_brightness": 0.6,
    "brightness_min": 0.7,
    "brightness_max": 1.3,
    "p_gauss_noise": 0.5,
    "noise_std_min": 5.0,
    "noise_std_max": 25.0,
    "p_blur": 0.5,
    "blur_kernel_choices": [3, 5, 7],
    "p_perspective": 0.5,  # 원근 변환 확률 원근 변환이란 이미지를 2D 평면에서 사각형의 네 꼭짓점을 다른 위치로 이동시켜 원근감을 주는 기법이다.
    "persp_ratio": 0.08,  # 원근 변환 비율 이미지 짧은 변의 8% 이내에서 변형
    "p_jpeg_artifact": 0.8,  # JPEG 압축을 걸 확률
    "jpeg_quality_min": 40,  # 40~90 사이에서 랜덤으로 고른 값으로 압축
    "jpeg_quality_max": 90,
    "p_cutout": 0.3,  # 30% 확률로 Cutout(일부 영역을 가려버리는 증강) 증강 적용
    "cutout_boxes": (1, 2),  # 몇 개의 가려진 사각형을 넣을지
    "cutout_size_ratio": (0.08, 0.2),  # 가려질 사각형의 크기 비율
    "max_transforms_per_image": 3,  # 이미지 하나에 적용할 수 있는 최대의 변환 수
}


# --------- 유틸 ---------------
def ensure_dir(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"'{p}'는 파일입니다. 디렉터리가 필요합니다.")
    p.mkdir(parents=True, exist_ok=True)


def list_images(
    folder: str,
) -> List[
    str
]:  # folder: str은 매개변수 folder은 문자열이여야 한다 그리고 -> List[str]은 함수의 반환값은 문자열들의 리스트다 라고 알려주는것
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def seeded_rng(base_seed: int, key: str) -> random.Random:
    h = int(hashlib.md5((str(base_seed) + key).encode("utf-8")).hexdigest(), 16) % (
        2**31
    )
    # base_seed를 str(문자열)로 변환하고 key(ex: img001)를 더한 후 utf-8로 encode한다 그거를 hashlib.md5에 넣어서 해시값을 구한다.
    # 그리고 그것을 hexdigest() 16진수 문자열로 변환한 후 int로 변환하고 2**31로 나눈 나머지를 구한다. int(..., 16) 은  16진수 문자열을 10진수 정수로 변환한다.
    rng = random.Random(h)  # 해시값을 시드로 하는 랜덤 객체 생성
    return rng


def imencode_jpeg_quality(
    image: np.ndarray, quality: int
) -> np.ndarray:  # ndarray는 N차원 배열 (N-dimensional array)
    # 보통 카메라를 이용해서 이미지를 추론할때 JPEG 압축된 이미지를 이용해서 추론하는 경우가 많음. 그래서 JPEG 압축 데이터를 증강시켜서 압축된 이미지에도 강한 모델 가능
    quality = int(
        np.clip(quality, 5, 100)
    )  # np.clip()은 5이하면 5로 만들고 100이상이면 100으로 만든다 ex: [0, 1, 2, 100, 105] -> [5, 5, 5, 100, 100]
    ok, enc = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )  # IMWRITE_JPEG_QUALITY는 JPEG 압축 품질을 설정하는 OpenCV 상수
    if not ok:  # 인코딩에 실패했을때 안전하게 원본 이미지 반환
        return image
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # 실제로 인코딩하고 디코딩 하면서 나타나는 데이터 손실을 반영하기 위해서 일부러 인코딩을 하고 디코딩을 한다.
    return dec if dec is not None else image


# ------------ 증강 함수들 ---------------
def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(
        img, cv2.COLOR_BGR2HSV
    )  # cv2.cvtColor : OpenCV 함수로, 이미지를 다른 색 공간(color space) 으로 변환 cv2.COLOR_BGR2HSV → BGR → HSV 변환.
    # BGR = (Blue, Green, Red) -> HSV = (Hue(색상), Saturation(채도), Value(명도))로 변환
    h, s, v = cv2.split(hsv)  # cv2.split : OpenCV 함수로, 이미지의 각 채널을 분리
    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    # v.astype(n.float32)는 v를 astype를 이용해서 n.float32로 타입 변환 시켜줌
    # factor은 밝기를 어느정도 조절할지 결정하느 인자 ex: fator = 0.7 -> 70%(더 어두움), factor = 1.3 -> 130% (더 밝게)
    # 한마디로 v(밝기)를 계산 용이하게 float32로 타입 변환을 시키고 factor를 곱해서 밝기를 변환시킨다음에 0이랑 255사이에 범위에 있어야되니까 np.clip으로 변환해서 그거를 다시 uint8(8비트 정수형)로 변환
    out = cv2.cvtColor(
        cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR
    )  # h, s, v로 분해한것을 다시 HSV이미지로 합치고 그것을 다시 BGR로 변환
    return out


def add_gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0.0, std, img.shape).astype(
        np.float32
    )  # np.random.normal은 정규분포(가우시안 분포)이고 인자는 평균, 표준편차, 크기
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(
        np.uint8
    )  # 원본 이미지에 노이즈를 추가하고 0~255 범위로 클리핑한 후 uint8로 변환
    return out


def random_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = (
        int(k) if int(k) % 2 == 1 else int(k) + 1
    )  # k가 float으로 들어올 수도 있으니까 int로 변환 그리고 k가 짝수이면 k+1로 만들어서 홀수로 만듬 (커널 크기는 홀수여야 함)
    # 3x3의 커널을 이용해서 원본 이미지를 곱하고 정규화 시켜서 블러 효과를 냄
    # ex)
    #  10  20  30  40     1 2 1
    #  50  60  70  80  X  2 4 2 / 16 ->  60  70
    #  90 100 110 120     1 2 1         100 110
    # 130 140 150 160
    return cv2.GaussianBlur(img, (k, k), 0)  # (k, k)는 커널 크기, 0은 표준편차


# [FIX] rng_local을 받아서 재현성 유지 + pts1 좌표 형태 (4,2)로 수정
def random_perspective(
    img: np.ndarray, ratio: float, rng_local: random.Random
) -> np.ndarray:
    h, w = img.shape[:2]  # 이미지의 높이와 너비 img = (높이, 너비, 채널수)
    off = int(min(h, w) * ratio)  # 최대 이동 가능한 픽셀 거리
    pts1 = np.float32(
        [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]
    )  # 원래 이미지의 네 꼭짓점 좌표 (4,2)
    # [FIX] 파일별 시드에 의해 결정되는 지터 사용
    jitter = np.array(
        [
            [rng_local.randint(-off, off), rng_local.randint(-off, off)]
            for _ in range(4)
        ],
        dtype=np.float32,
    )
    pts2 = pts1 + jitter  # 네 꼭짓점을 랜덤하게 이동
    pts2[:, 0] = np.clip(
        pts2[:, 0], 0, w - 1
    )  # x좌표 클리핑 [:, 0]은 모든행의 0번째 열 = x 좌표
    pts2[:, 1] = np.clip(
        pts2[:, 1], 0, h - 1
    )  # y좌표 클리핑 [:, 1]은 모든행의 1번째 열 = y 좌표
    M = cv2.getPerspectiveTransform(
        pts1, pts2
    )  # 변환 행렬 계산 pts1 : 원본 네 꼭짓점 좌표, pts2 : 이동된 네 꼭짓점 좌표
    return cv2.warpPerspective(
        img, M, (w, h), borderMode=cv2.BORDER_REPLICATE
    )  # 위에서 만든 M의 변환 행렬을 이용해서 이미지 변환 borderMode는 빈공간을 어떻게 채울지 결정 cv2.BORDER_REPLICATE는 가장자리를 복제


# [FIX] rng_local을 선택 인자로 받아서 재현성 유지
def random_cutout(
    img: np.ndarray,
    box_range=(1, 2),
    size_ratio=(0.1, 0.2),
    rng_local: random.Random | None = None,
) -> np.ndarray:  # cutout증강이란 이미지 일부분을 사각형으로 잘라서 변형
    r = rng_local or random  # [FIX] 파일별 RNG 우선 사용
    h, w = img.shape[:2]
    out = img.copy()
    n_boxes = r.randint(box_range[0], box_range[1])
    # 여기서 사용하는 random.randint는 난수 하나를 생성하는 코드로 import random에서 사용하는거 그리고 범위가 box_range[1]도 포함.
    # 그리고 이전에 사용했던 rng.integers는 numpy에서 사용하는 generator객체로 import random이 아니고 numpy를 import해야 사용 가능 그리고 box_range[1]은 포함 X
    for _ in range(
        n_boxes
    ):  # _는 i랑 같이 변수이지만 변수를 딱히 쓰고 싶지 않을때 사용하는 변수명
        bw = int(
            r.uniform(size_ratio[0], size_ratio[1]) * w
        )  # random.uniform은 size_ratio[0] <=, size_ratio[1] >= 사이의 균일 분포 실수를 생성
        bh = int(r.uniform(size_ratio[0], size_ratio[1]) * h)
        x1 = r.randint(0, max(0, w - bw))
        y1 = r.randint(0, max(0, h - bh))
        out[y1 : y1 + bh, x1 : x1 + bw] = (
            0  # 검정 패치. 필요 시 평균색/랜덤색으로 변경 가능.
        )
    return out


# ------------------------------
# 3) 한 장에 대해 "무작위로 몇 개" 변형 선택
# ------------------------------
def apply_random_transforms(  # 이미지에 어떤 증강을 어떤 세기로 적용할지 리스트로 정리하고 그 리스트를 섞어서 3개만 골라서 적용
    img: np.ndarray,
    cfg: dict,
    rng_local: random.Random,  # cfg : 증강에 쓰이는 설정값 ex) p_brightness, brightness_min/max 등등
) -> np.ndarray:
    ops = []

    if (
        rng_local.random() < cfg["p_brightness"]
    ):  # 밝기 증강을 적용할 확률 : cfg["p_brightness"]
        factor = rng_local.uniform(cfg["brightness_min"], cfg["brightness_max"])
        ops.append(lambda im: adjust_brightness(im, factor))

    if rng_local.random() < cfg["p_gauss_noise"]:
        std = rng_local.uniform(cfg["noise_std_min"], cfg["noise_std_max"])
        ops.append(lambda im: add_gaussian_noise(im, std))

    if rng_local.random() < cfg["p_blur"]:
        k = rng_local.choice(cfg["blur_kernel_choices"])
        ops.append(lambda im: random_blur(im, k))

    if rng_local.random() < cfg["p_perspective"]:
        ops.append(
            lambda im: random_perspective(im, cfg["persp_ratio"], rng_local)
        )  # [FIX] 시그니처 일치

    if rng_local.random() < cfg["p_jpeg_artifact"]:
        q = rng_local.randint(cfg["jpeg_quality_min"], cfg["jpeg_quality_max"])
        ops.append(lambda im: imencode_jpeg_quality(im, q))

    if rng_local.random() < cfg["p_cutout"]:
        ops.append(
            lambda im: random_cutout(
                im, cfg["cutout_boxes"], cfg["cutout_size_ratio"], rng_local
            )
        )  # [FIX] 시그니처 일치

    # 한 이미지에 동시에 적용할 최대 개수 제한
    rng_local.shuffle(ops)
    k = rng_local.randint(1, max(1, min(len(ops), cfg["max_transforms_per_image"])))
    out = img.copy()
    for fn in ops[:k]:
        out = fn(out)
    return out


# ----------- 메인 증강 함수 -----------------
def augment_and_save(
    source_dir: str,
    output_dir: str,
    num_augmentations: int = 4,
    base_seed: int = 42,
    keep_original: bool = True,
    skip_if_aug_exists: bool = True,  # 그대로 둠(철자 그대로). 필요하면 exists로 바꿔도 됨.
    cfg: dict = AUG_CONFIG,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.is_dir():
        print(f"[!] Source not found: {source_dir}")
        return

    ensure_dir(output_dir)

    files = list_images(source_dir)
    if not files:
        print(f"[!] No images in: {source_dir}")
        return

    if skip_if_aug_exists and any(
        "_aug" in f for f in os.listdir(output_dir)
    ):  # skip_if_aug_exists는 같은게 있을때 스킵할지 여부인데 예를들어 _aug가 이미 있는데 다른 시드라던가 정보를 이용해서 시도해보고 싶을때 이미 데이터가 있어서 덮어쓰기 하고 싶을때 False로 설정
        print(
            f"[i] Found augmented files in {output_dir}.Skipping to avoid duplicates."
        )
        return

    same_io = (
        source_dir.resolve() == output_dir.resolve()
    )  # Path.resolve()는 절대경로로 바꿔서 .라던가 /라던가.. 상대 요소를 없앰

    print(f"[i] Augmenting {len(files)} images from {source_dir} -> {output_dir}")
    for fn in files:
        src_path = source_dir / fn
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[!] Cannot read: {src_path}")  # [FIX] 오타 수정(readL -> read)
            continue

        stem, ext = os.path.splitext(
            fn
        )  # os.path.splitext는 이름과 확장자를 분류하는 부분
        ext = ext if ext else ".png"  # 만약에 확장자가 없으면 .png를 붙인다

        if keep_original:  # True 라면 증강본 저장과 별개로 원본도 output_dir에 복사
            dst_ori = output_dir / f"{stem}{ext}"  # 원본을 저장할 목적지 경로
            if (
                not same_io and not dst_ori.exists()
            ):  # 입력 폴더랑 출력 폴더가 동일한지(같으면 이미 원본이 폴더에 있으니까 불필요한 복사 방지) + 목적지에 동일한 파일이 존재하지 않을때
                cv2.imwrite(str(dst_ori), img)  # 원본 이미지를 출력 폴던에 쓰기

        rng_local = seeded_rng(base_seed, stem)  # 객체 생성

        for i in range(num_augmentations):
            np.random.seed(
                rng_local.randint(0, 1_000_000)
            )  # rng_local에서 뽑은 난수를 Numpy 전역시드에 대입
            random.seed(
                rng_local.randint(0, 1_000_000)
            )  # rng_local에서 뽑은 난수를 파이썬 전역시드에 대입

            aug = apply_random_transforms(img, cfg, rng_local)
            out_name = f"{stem}_aug{i}{ext.lower()}"
            out_path = output_dir / out_name
            cv2.imwrite(
                str(out_path), aug
            )  # [FIX] 오타 수정: str(out_path).aug -> (str(out_path), aug)

    print(
        f"[✓] Done. Saved to: {output_dir}"
    )  # [FIX] 루프 밖으로 이동(전체 완료 후 1회)


# ----------------- 예시 실행 -------------------
if (
    __name__ == "__main__"
):  # 이 파일이 “직접 실행된 스크립트”인지, “다른 파일에서 import된 모듈”인지 구분해서, 특정 코드를 실행할지 말지 결정하는 표준 관용구
    # 권장: 원본과 증강본을 분리해서 관리
    # 원본과 같은 폴더를 쓰고 싶다면 아래 두 줄에서 *_aug를 원본 폴더로 바꾸면 됨.
    augment_and_save(
        "Data/True/train",
        "Data/True_aug",
        num_augmentations=4,
        base_seed=42,
        keep_original=True,
    )

    augment_and_save(
        "Data/False/train",
        "Data/False_aug",
        num_augmentations=4,
        base_seed=42,
        keep_original=True,
    )
