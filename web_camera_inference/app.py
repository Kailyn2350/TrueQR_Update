import base64
import io
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory, Response

# --- Flask App Configuration ---
app = Flask(__name__, static_folder=".")  # 백엔드 프레임워크 Python
model = None  # 모델을 로드하기 위해서 미리 선언

# --- Model Configuration (Updated for Moiré Pattern CNN) ---
IMG_SIZE = 224
CLASS_NAMES = ["False", "True"]


def load_new_model():
    """Load the trained Keras model for Moiré pattern detection."""
    global model  # 이것을 안하고 그냥 model을 사용하면 지역변수용 model을 생성해버리니까 전역변수의 model을 사용한다는 의미에서 global을 이용
    try:
        # Construct the path to the model relative to this script's location
        script_dir = os.path.dirname(
            os.path.abspath(__file__)
        )  # os.path.abspath는 __file__의 절대경로를 반환, __file__은 실행하고 있는 .py의 경로를 반환, os.path.dirname은 상위 디렉토리를 반환
        # 한마디로 C:\Users\seoks\Desktop\Util\TrueQR_Update\web_camera_inference\app.py 를 abspath로 가져오고 C:\Users\seoks\Desktop\Util\TrueQR_Update\web_camera_inference 를 dirname을 통해서 script_dir에 저장
        project_root = os.path.dirname(
            script_dir
        )  # C:\Users\seoks\Desktop\Util\TrueQR_Update로 한번더 이동 루트 폴더 위치로 이동해서 모델이 들어있는 results_advanced를 찾기 위해서
        # Updated model path
        model_path = (
            os.path.join(  # os.path.join은 경로를 아키텍처에 맞게 합쳐주는 함수
                project_root, "results_advanced", "qr_attention_model.keras"
            )
        )

        if not os.path.exists(model_path):  # 해당 경로에 모델이 존재하는지 확인
            print(f"[ERROR] Model file not found at: {model_path}")
            return

        model = tf.keras.models.load_model(model_path)  # TensorFlow Keras 모델 로드
        print(f"[INFO] Loaded Moiré pattern model from {model_path}")
    except Exception as e:
        print(f"[ERROR] Could not load Keras model: {e}")


@app.before_request  # 모든 요청이 Flask의 라우트 함수로 들어가기 직전에 실행
def log_request_info():
    """Log incoming request paths."""
    if (
        request.path != "/favicon.ico"
    ):  # 웹을 열었을때 아이콘을 표시하기 위한 요청의 경우는 로그 필요 없으니까 무시
        print(
            f"[REQUEST] Path: {request.path}, Method: {request.method}"
        )  # 요청 URL 경로(path)와 HTTP 메서드(GET, POST 등)표시


@app.route("/")
def index():
    """Serve the main index.html page."""
    return send_from_directory(
        ".", "index.html"
    )  # Flask 내장 함수로, 특정 디렉토리 안의 파일을 클라이언트에게 보내주는 함수 ex : send_from_directory(directory, filename)


@app.route(
    "/favicon.ico"
)  # 아이콘을 표시하기 위한 라우터. 웹페이지에 접속하면 사용자가 자동으로 요청하는 아이콘 파일
def favicon():
    """Handle favicon requests."""
    return Response(
        status=204
    )  # 자동으로 아이콘을 요청하는데 나는 따로 아이콘을 설정하지 않았으니까 404가 뜨는거를 방지하기 위해서 204 No Content 상태 코드를 반환


@app.route(
    "/predict", methods=["POST"]
)  # GET은 URL에 간단한 파라미터를 붙여서 읽기 전용 요청 / POST는 요청 바디(body)에 JSON·파일 같은 큰 데이터를 담아 보낼 때 주로 사용
# 여기서 이미지는 0과 1로 이루어진 바이너리 데이터인데 그것을 그대로 JSON을 이용해서 보내면 파싱에러가 남 예를들어 0과1 이외에도 \x00 같은게 들어가면 파싱에러가 나기 때문에 Base64로 인코딩해서 JSON에 넣어서 보냄
def predict_image():
    """Receive an image frame, run inference, and return the result."""
    if model is None:
        return jsonify({"result": "Error: Model not loaded.", "detail": {}}), 500

    try:
        # --- 1. Decode Image --- # 클라이언트가 base64로 인코딩한 이미지를 디코딩
        data = (
            request.get_json()
        )  # Flask의 내장 메소드로 get_json은 클라이언트가 보낸 HTTP 요청 request에서 JSON을 추출
        image_data = data[
            "image"
        ]  # app.js를 이용해 클라이언트가 보낸 JSON의 "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."에서 image 부분을 가져옴
        header, encoded = image_data.split(
            ",", 1
        )  # ,콤마를 기준으로 1번 분리 str.split(sep, maxsplit) 사용법 한마디로 헤더 부분 data:image/jpeg;base64 이거랑 인코딩된 데이터 부분을 분리하기 위해서
        binary_data = base64.b64decode(
            encoded
        )  # 이미지 파일의 원본 바이트열 ex : b'\xff\xd8\xff\xe0\x00\x10JFIF...\xff\xd9' 압축된 데이터
        image_np = np.frombuffer(
            binary_data, dtype=np.uint8
        )  # 압축된 데이터를 정수 배열로 변환. np.frombuffer는 바이트 데이터를 NumPy 배열로 변환하는 함수, dtype=np.uint8은 0~255 사이의 정수로 표현되는 이미지 데이터를 의미

        # Decode image in grayscale as required by the model
        img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (
                jsonify({"result": "Error: Image decoding failed.", "detail": {}}),
                400,
            )

        # --- 2. Preprocess Image ---
        # Resize to model's expected input size
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Reshape for model input, normalize, and create a batch
        img_array = np.array(img_resized).reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0
        img_batch = np.expand_dims(img_array, 0)

        # --- 3. Run Inference ---
        prediction = model.predict(img_batch)
        score = float(prediction[0][0])

        # --- 4. Format Response ---
        # Use 0.5 as the threshold as per train_cnn.py
        is_true = score > 0.7
        result_text = CLASS_NAMES[1] if is_true else CLASS_NAMES[0]

        detail = {
            "score": f"{score:.4f}",
            "prediction": result_text,
            "is_genuine": bool(is_true),
        }

        return jsonify({"result": result_text, "detail": detail})

    except Exception as e:
        print(f"[ERROR] An exception occurred in /predict: {e}")
        return jsonify({"result": "Server Error", "detail": str(e)}), 500


@app.route("/<path:path>")
def serve_file(path):
    """Serve other static files like CSS and JS."""
    return send_from_directory(".", path)


if __name__ == "__main__":
    load_new_model()
    # Running with debug=False is recommended for stability
    # as debug mode can sometimes cause issues with model loading.
    app.run(host="0.0.0.0", port=8000, debug=False)
