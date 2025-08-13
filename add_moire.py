# 운영체제의 파일 및 디렉토리 관련 기능을 사용하기 위해 'os' 모듈을 불러옵니다.
import os

# 이미지 처리 라이브러리인 Pillow에서 이미지 생성/열기(Image)와
# 이미지에 그림 그리기(ImageDraw) 기능을 불러옵니다.
from PIL import Image, ImageDraw

# --- 디렉토리 설정 ---
# 원본 이미지들이 들어있는 폴더의 이름을 'originals'로 지정합니다.
source_dir = "originals"
# 결과 이미지를 저장할 폴더의 이름을 'moire_images_light'로 지정합니다.
output_dir = "moire_images_light"

# --- 출력 폴더 생성 ---
# 만약 'moire_images_light' 폴더가 존재하지 않으면,
if not os.path.exists(output_dir):
    # 새로운 폴더를 생성합니다.
    os.makedirs(output_dir)

# --- 처리할 이미지 파일 목록 가져오기 ---
# 'originals' 폴더 내의 모든 파일 목록을 가져와서,
# 파일 이름이 '.png'로 끝나는 파일만 골라 리스트로 만듭니다.
image_files = [f for f in os.listdir(source_dir) if f.endswith(".png")]

# --- 각 이미지에 대해 패턴 적용 작업 반복 ---
# 위에서 만든 이미지 파일 리스트를 하나씩 순회합니다.
for filename in image_files:
    # --- 원본 이미지 열기 및 준비 ---
    # 원본 폴더 경로와 파일 이름을 합쳐 전체 경로를 만듭니다. (예: 'originals/qrcode.png')
    img_path = os.path.join(source_dir, filename)
    # 이미지 파일을 열고, 투명도(Alpha)를 다룰 수 있는 'RGBA' 모드로 변환합니다.
    original_image = Image.open(img_path).convert("RGBA")
    # 원본 이미지의 가로(width)와 세로(height) 크기를 변수에 저장합니다.
    width, height = original_image.size

    # --- 첫 번째 패턴 (수직선) 생성 ---
    # 첫 번째 패턴을 그릴 완전히 투명한 새 도화지를 만듭니다.
    pattern1 = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    # 이 'pattern1' 도화지에 그림을 그릴 수 있는 도구를 준비합니다.
    p1_draw = ImageDraw.Draw(pattern1)
    # 이미지의 왼쪽 끝(x=0)부터 오른쪽 끝까지 8픽셀 간격으로 반복합니다.
    for i in range(0, width, 8):
        # 현재 위치(i)에 위에서 아래로 수직선을 그립니다.
        # fill=(0, 0, 0, 64): 검은색(0,0,0)을 64의 투명도(알파값, 255가 불투명)로 그립니다.
        # 즉, 매우 옅은 반투명 검은색 선입니다.
        p1_draw.line([(i, 0), (i, height)], fill=(0, 0, 0, 64), width=1)

    # --- 두 번째 패턴 (회전된 수직선) 생성 ---
    # 두 번째 패턴을 그릴 또 다른 투명한 도화지를 만듭니다.
    pattern2 = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    # 'pattern2' 도화지에 그림을 그릴 도구를 준비합니다.
    p2_draw = ImageDraw.Draw(pattern2)
    # 첫 번째 패턴과 똑같이 8픽셀 간격의 반투명 수직선들을 그립니다.
    for i in range(0, width, 8):
        p2_draw.line([(i, 0), (i, height)], fill=(0, 0, 0, 64), width=1)

    # 방금 만든 두 번째 패턴을 5도 회전시킵니다.
    # expand=True는 이미지가 회전하면서 잘리지 않도록 캔버스 크기를 자동으로 확장하는 옵션입니다.
    pattern2 = pattern2.rotate(5, expand=True)

    # --- 회전된 이미지를 원래 크기로 잘라내기 ---
    # 회전 후 확장된 이미지의 새로운 가로, 세로 크기를 가져옵니다.
    w, h = pattern2.size
    # 원본 크기로 잘라내기 위해, 잘라낼 영역의 왼쪽 위 시작 좌표(x, y)를 계산합니다.
    x = (w - width) / 2
    y = (h - height) / 2
    # 계산된 좌표를 이용해, 회전된 이미지의 중앙 부분을 원본 이미지와 똑같은 크기로 잘라냅니다.
    pattern2 = pattern2.crop((x, y, x + width, y + height))

    # --- 두 패턴을 하나로 합치기 ---
    # 첫 번째 패턴(pattern1) 이미지 위에, 회전된 두 번째 패턴(pattern2)을 겹칩니다.
    # 두 개의 반투명한 격자가 겹쳐지며 미세한 간섭 무늬가 생깁니다.
    moire_overlay = Image.alpha_composite(pattern1, pattern2)

    # --- 원본 이미지와 모아레 패턴 합성 ---
    # 원본 QR 코드 이미지 위에, 방금 합친 모아레 패턴 오버레이를 겹쳐 최종 이미지를 만듭니다.
    final_image = Image.alpha_composite(original_image, moire_overlay)

    # --- 최종 이미지 저장 ---
    # 저장할 파일의 전체 경로를 만듭니다. (예: 'moire_images_light/qrcode.png')
    output_path = os.path.join(output_dir, filename)
    # 최종 결과 이미지를 파일로 저장합니다.
    final_image.save(output_path)

# --- 작업 완료 메시지 출력 ---
# 모든 작업이 끝난 후, 몇 개의 이미지를 처리했고 어디에 저장했는지 알려주는 메시지를 출력합니다.
print(f"Processed {len(image_files)} images and saved them to '{output_dir}'.")
