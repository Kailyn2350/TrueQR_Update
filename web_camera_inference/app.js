// ========= 기존 요소 참조 =========
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const banner = document.getElementById("banner");
const debugInfo = document.createElement("div");
document.body.appendChild(debugInfo);
const ctx = canvas.getContext("2d");
let isRequesting = false;

debugInfo.style.position = "absolute";
debugInfo.style.bottom = "100px";
debugInfo.style.left = "50%";
debugInfo.style.transform = "translateX(-50%)";
debugInfo.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
debugInfo.style.padding = "10px";
debugInfo.style.borderRadius = "5px";
debugInfo.style.fontSize = "0.9em";
debugInfo.style.whiteSpace = "pre";
debugInfo.style.fontFamily = "monospace";
debugInfo.style.display = "none";

// ========= 추가: Flip / Torch 버튼 참조 =========
const btnTorch = document.getElementById("toggleTorch");
const btnFlip = document.getElementById("switchCam");

// ========= 상태 =========
let currentStream = null;
let currentVideoTrack = null;
let devices = [];             // videoinput 목록
let currentDeviceIndex = -1;  // 현재 사용 중인 카메라 인덱스
let torchOn = false;

const PREDICT_URL = "/predict";

// ========= 공통 유틸 =========
function setBanner(msg) {
  if (banner) banner.textContent = msg;
}

async function stopStream() {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
    currentVideoTrack = null;
  }
}

async function listVideoInputs() {
  const all = await navigator.mediaDevices.enumerateDevices();
  devices = all.filter(d => d.kind === "videoinput");
}

function guessBackCameraIndex() {
  // 라벨에 back/rear가 있으면 우선 선택, 없으면 첫 번째 장치를 기본값으로 사용
  const lower = devices.map(d => (d.label || "").toLowerCase());
  let idx = lower.findIndex(l => l.includes("back") || l.includes("rear"));
  if (idx === -1) idx = 0; // "back" 카메라를 찾지 못하면 첫 번째 카메라를 기본값으로 설정
  return idx;
}

function updateTorchVisibility() {
  try {
    const caps = currentVideoTrack?.getCapabilities?.();
    const supported = !!(caps && "torch" in caps && caps.torch);
    // 지원 안 되면 버튼 숨김
    if (btnTorch) btnTorch.style.display = supported ? "inline-flex" : "none";
  } catch {
    if (btnTorch) btnTorch.style.display = "none";
  }
}

// ========= 스트림 시작 (장치 인덱스로) =========
async function startStreamByIndex(idx) {
  await stopStream();
  currentDeviceIndex = idx;

  const deviceId = devices[idx]?.deviceId;
  const constraints = deviceId
    ? { video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false }
    : { video: { facingMode: { exact: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    // facingMode 등의 제약 실패 시 완화
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }

  currentStream = stream;
  currentVideoTrack = currentStream.getVideoTracks()[0];
  video.srcObject = currentStream;

  // 메타 갱신 후 캔버스 사이즈 세팅
  await new Promise(resolve => {
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      resolve();
    };
  });

  // 배너 갱신
  setBanner(devices[currentDeviceIndex]?.label || "Camera ready");
  updateTorchVisibility();
}

// ========= 초기 카메라 준비 (권한 부여 & 장치 라벨 노출 유도) =========
async function initCamera() {
  try {
    // 권한 요청으로 라벨 노출 유도 (일부 브라우저)
    try {
      const temp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      temp.getTracks().forEach(t => t.stop());
    } catch { /* 무시 */ }

    await listVideoInputs();

    if (devices.length === 0) {
      console.error("No camera devices found");
      banner.textContent = "Could not access the camera.";
      banner.style.backgroundColor = "red";
      // 버튼 숨김
      if (btnFlip) btnFlip.style.display = "none";
      if (btnTorch) btnTorch.style.display = "none";
      return;
    }

    const idx = guessBackCameraIndex();
    await startStreamByIndex(idx);

  } catch (err) {
    console.error("Camera Error:", err);
    banner.textContent = "Could not access the camera.";
    banner.style.backgroundColor = "red";
  }
}

// ========= Torch 토글 =========
async function toggleTorch() {
  if (!currentVideoTrack) return;

  const caps = currentVideoTrack.getCapabilities?.();
  if (!(caps && "torch" in caps && caps.torch)) {
    setBanner("Torch not supported on this device/browser");
    return;
  }

  torchOn = !torchOn;
  try {
    // 일부 구현은 advanced 배열 필요
    await currentVideoTrack.applyConstraints({ advanced: [{ torch: torchOn }] });
    btnTorch?.setAttribute("aria-pressed", String(torchOn));
    setBanner(torchOn ? "Torch ON" : "Torch OFF");
  } catch (e) {
    // 실패 시 되돌리기
    torchOn = !torchOn;
    btnTorch?.setAttribute("aria-pressed", String(torchOn));
    setBanner("Failed to toggle torch");
  }
}

// ========= 카메라 전환 =========
async function switchCamera() {
  if (devices.length <= 1) {
    setBanner("Only one camera detected");
    return;
  }
  const next = (currentDeviceIndex + 1) % devices.length;
  torchOn = false; // 전환 시 torch 초기화
  btnTorch?.setAttribute("aria-pressed", "false");
  await startStreamByIndex(next);
}

// ========= 기존 마스크/렌더/검증 로직 유지 =========
function drawMask() {
  const w = canvas.width;
  const h = canvas.height;
  const boxSize = Math.min(w, h) * 0.7;
  const left = (w - boxSize) / 2;
  const top = (h - boxSize) / 2;

  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillRect(left, top, boxSize, boxSize);
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 3;
  ctx.strokeRect(left, top, boxSize, boxSize);
  ctx.restore();
}

async function verifyFrame() {
  if (isRequesting || !video.srcObject) return;
  isRequesting = true;

  const vw = video.videoWidth, vh = video.videoHeight;
  const box = Math.min(vw, vh) * 0.7;
  const sx = (vw - box) / 2, sy = (vh - box) / 2;

  const crop = document.createElement("canvas");
  crop.width = 512; crop.height = 512;
  const cctx = crop.getContext("2d");
  cctx.drawImage(video, sx, sy, box, box, 0, 0, 512, 512);

  const jpegDataUrl = crop.toDataURL("image/jpeg", 0.8);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);

  try {
    const res = await fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: jpegDataUrl }),
      signal: controller.signal,
    });

    const bodyText = await res.text().catch(() => "");
    if (!res.ok) throw new Error(`Server ${res.status}: ${bodyText || "no body"}`);
    const data = bodyText ? JSON.parse(bodyText) : {};
    updateBanner(data.result, data.detail);
  } catch (e) {
    console.error("Prediction failed:", e);
    updateBanner("Prediction Error", {});
  } finally {
    clearTimeout(timer);
    setTimeout(() => { isRequesting = false; }, 500); // ~2fps 제한
  }
}

function updateBanner(result, detail) {
  let bannerText = result;
  if (detail && detail.score) bannerText = `${result} (Score: ${detail.score})`;
  banner.textContent = bannerText;

  if (detail && detail.is_genuine === true) {
    banner.className = "detect-banner genuine";
  } else if (detail && detail.is_genuine === false) {
    banner.className = "detect-banner fake";
  } else {
    banner.className = "detect-banner";
  }
  debugInfo.style.display = "none";
}

function renderLoop() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  drawMask();
  requestAnimationFrame(renderLoop);
}

// ========= 초기화 =========
async function main() {
  // 보안 컨텍스트(HTTPS 또는 localhost) 권장
  if (!("mediaDevices" in navigator) || !navigator.mediaDevices.getUserMedia) {
    setBanner("getUserMedia not supported");
    if (btnFlip) btnFlip.style.display = "none";
    if (btnTorch) btnTorch.style.display = "none";
    return;
  }

  await initCamera();
  renderLoop();
  setInterval(verifyFrame, 1000);

  // 버튼 핸들러
  btnFlip?.addEventListener("click", switchCamera);
  btnTorch?.addEventListener("click", toggleTorch);

  // 카메라가 1개면 Flip 숨김
  if (devices.length <= 1 && btnFlip) btnFlip.style.display = "none";
}
main();
