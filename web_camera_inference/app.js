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

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });
    video.srcObject = stream;
    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });
  } catch (err) {
    console.error("Camera Error:", err);
    banner.textContent = "Could not access the camera.";
    banner.style.backgroundColor = "red";
  }
}

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

const VERIFY_URL = "/verify"; // 필요하면 절대경로 사용

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

  // PNG → JPEG로 변경: 페이로드 대폭 감소
  const jpegDataUrl = crop.toDataURL("image/jpeg", 0.6);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);

  try {
    const res = await fetch(VERIFY_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: jpegDataUrl }), // 서버가 req.body.image를 기대해야 함
      signal: controller.signal,
    });

    const bodyText = await res.text().catch(() => "");
    if (!res.ok) throw new Error(`Server ${res.status}: ${bodyText || "no body"}`);
    const data = bodyText ? JSON.parse(bodyText) : {};
    updateBanner(data.result, data.detail);
  } catch (e) {
    console.error("Verification failed:", e);
    updateBanner("Verification Error", {});
  } finally {
    clearTimeout(timer);
    setTimeout(() => { isRequesting = false; }, 400); // 2~3fps 정도로 제한
  }
}

function updateBanner(result, detail) {
  // Default text is just the result (e.g., "Error")
  let bannerText = result;

  // If we have a score in the detail object, format it.
  if (detail && typeof detail.score === 'number') {
    const percentage = (detail.score * 100).toFixed(1);
    bannerText = `${result} (${percentage}%)`;
  }

  banner.textContent = bannerText;

  if (result.includes("GENUINE")) {
    banner.className = "detect-banner genuine";
  } else if (result.includes("COPY")) {
    banner.className = "detect-banner fake";
  } else {
    banner.className = "detect-banner";
  }

  // The old debugInfo is not needed anymore with the new model
  debugInfo.style.display = "none";
}

function renderLoop() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  drawMask();
  requestAnimationFrame(renderLoop);
}

async function main() {
  await initCamera();
  renderLoop();
  setInterval(verifyFrame, 1000);
}

main();
