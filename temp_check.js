
    import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';

    const VIDEO_WIDTH = 1280;
    const VIDEO_HEIGHT = 720;
    const modelAssetPath = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task';
    const assetsPath = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';

    const video = document.getElementById('video');
    const viewport = document.getElementById('viewport');
    const hiddenCanvas = document.getElementById('hiddenCanvas');
    const thermalCanvas = document.getElementById('thermalCanvas');
    const ctx = viewport.getContext('2d');
    const hiddenCtx = hiddenCanvas.getContext('2d');
    const thermalCtx = thermalCanvas.getContext('2d');

    const hudLeft = document.getElementById('hud-left');
    const hudRight = document.getElementById('hud-right');
    const hudState = document.getElementById('hud-state');
    const hudView = document.getElementById('hud-view');
    const toggleBtn = document.getElementById('toggleBtn');
    const boot = document.getElementById('boot');
    const enterBtn = document.getElementById('enterBtn');

    let handLandmarker = null;
    let cameraStream = null;
    let running = false;
    let animationHandle = null;
    let seedHex = generateSeed();
    let hudJson = {
      seed: seedHex,
      area: '00%',
      tmax: '28°C',
      state: 'IDLE',
      view: 'SCAN',
      dot: '#7f8fa4'
    };
    let lastBox = null;
    let motionScalar = 0;

    const shapeState = {
      center: { x: VIDEO_WIDTH / 2, y: VIDEO_HEIGHT / 2 },
      angle: 0,
      width: VIDEO_WIDTH * 0.45,
      height: VIDEO_HEIGHT * 0.45
    };

    enterBtn.addEventListener('click', async () => {
      boot.style.display = 'none';
      await initCameraPipeline();
    });

    toggleBtn.addEventListener('click', async () => {
      if (running) {
        stopCamera();
      } else {
        await initCameraPipeline();
      }
    });

    async function initCameraPipeline() {
      toggleBtn.disabled = true;
      if (!cameraStream) {
        cameraStream = await openCamera();
      }

      if (!handLandmarker) {
        await loadMediaPipe();
      }

      if (!running) {
        running = true;
        toggleBtn.textContent = 'Stop Camera';
        animateFrame();
      }
      toggleBtn.disabled = false;
    }

    async function openCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT }, audio: false });
      video.srcObject = stream;
      await video.play();
      return stream;
    }

    function stopCamera() {
      running = false;
      toggleBtn.textContent = 'Start Camera';
      if (animationHandle) {
        cancelAnimationFrame(animationHandle);
        animationHandle = null;
      }
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
      }
      handLandmarker = handLandmarker;
      ctx.fillStyle = '#040c18';
      ctx.fillRect(0, 0, viewport.width, viewport.height);
      hudJson.state = 'IDLE';
      hudJson.dot = '#7f8fa4';
      updateHudOverlay();
    }

    async function loadMediaPipe() {
      const vision = await FilesetResolver.forVisionTasks(assetsPath);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.55,
        minTrackingConfidence: 0.55,
        minHandPresenceConfidence: 0.5
      });
    }

    function animateFrame() {
      if (!running) {
        return;
      }

      drawFrame()
        .catch((error) => {
          console.warn('Frame processing error:', error);
        })
        .finally(() => {
          animationHandle = requestAnimationFrame(animateFrame);
        });
    }

    async function drawFrame() {
      hiddenCanvas.width = VIDEO_WIDTH;
      hiddenCanvas.height = VIDEO_HEIGHT;
      thermalCanvas.width = VIDEO_WIDTH;
      thermalCanvas.height = VIDEO_HEIGHT;

      hiddenCtx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
      const imageData = hiddenCtx.getImageData(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
      thermalCtx.putImageData(applyThermalColormap(imageData), 0, 0);

      ctx.clearRect(0, 0, viewport.width, viewport.height);
      ctx.drawImage(video, 0, 0, viewport.width, viewport.height);

      const result = await handLandmarker.detectForVideo(video, performance.now());
      const landmarks = result?.landmarks?.[0] ?? [];
      const boxInfo = landmarks.length ? computeRotatedBoundingBox(landmarks) : null;

      if (boxInfo) {
        lastBox = boxInfo;
        motionScalar = computeMotion(boxInfo);
        drawThermalScanBox(boxInfo);
      } else {
        motionScalar = Math.max(0, motionScalar * 0.92 - 0.02);
      }

      updateHudData(boxInfo);
      drawHudOverlay(boxInfo);
    }

    function computeRotatedBoundingBox(landmarks) {
      const coords = landmarks.map((pt) => ({ x: pt.x * VIDEO_WIDTH, y: pt.y * VIDEO_HEIGHT }));
      const xValues = coords.map((p) => p.x);
      const yValues = coords.map((p) => p.y);
      const minX = Math.min(...xValues);
      const maxX = Math.max(...xValues);
      const minY = Math.min(...yValues);
      const maxY = Math.max(...yValues);
      const width = Math.max((maxX - minX) * 1.3, 120);
      const height = Math.max((maxY - minY) * 1.6, 120);
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const direction = getDirectionVector(coords);
      const angle = Math.atan2(direction.y, direction.x);

      return { cx, cy, width, height, angle };
    }

    function getDirectionVector(coords) {
      if (coords.length < 2) {
        return { x: 1, y: 0 };
      }
      const wrist = coords[0];
      const fingertip = coords[9] ?? coords[1];
      return {
        x: fingertip.x - wrist.x,
        y: fingertip.y - wrist.y
      };
    }

    function computeMotion(boxInfo) {
      if (!lastBox) {
        return 0.08;
      }
      const dx = Math.abs(boxInfo.cx - lastBox.cx);
      const dy = Math.abs(boxInfo.cy - lastBox.cy);
      return Math.min(1, Math.max(0, (dx + dy) / 40));
    }

    function drawThermalScanBox(boxInfo) {
      const scaled = scaleBoxToViewport(boxInfo);
      ctx.save();
      ctx.translate(scaled.cx, scaled.cy);
      ctx.rotate(scaled.angle);
      ctx.beginPath();
      ctx.rect(-scaled.width / 2, -scaled.height / 2, scaled.width, scaled.height);
      ctx.clip();
      ctx.globalAlpha = 0.96;
      ctx.drawImage(thermalCanvas, 0, 0, viewport.width, viewport.height);
      ctx.globalAlpha = 1;
      ctx.restore();
      drawScanFrame(scaled);
    }

    function drawScanFrame(box) {
      const border = 2;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.92)';
      ctx.lineWidth = border;
      ctx.translate(box.cx, box.cy);
      ctx.rotate(box.angle);
      ctx.strokeRect(-box.width / 2, -box.height / 2, box.width, box.height);
      drawCornerBracket(-box.width / 2, -box.height / 2, 22, 18);
      drawCornerBracket(box.width / 2, -box.height / 2, -22, 18);
      drawCornerBracket(-box.width / 2, box.height / 2, 22, -18);
      drawCornerBracket(box.width / 2, box.height / 2, -22, -18);
      ctx.restore();
    }

    function drawCornerBracket(x, y, offsetX, offsetY) {
      ctx.beginPath();
      ctx.moveTo(x + Math.sign(offsetX) * 4, y);
      ctx.lineTo(x + offsetX, y);
      ctx.lineTo(x + offsetX, y + offsetY);
      ctx.stroke();
    }

    function scaleBoxToViewport(box) {
      const scaleX = viewport.width / VIDEO_WIDTH;
      const scaleY = viewport.height / VIDEO_HEIGHT;
      return {
        cx: box.cx * scaleX,
        cy: box.cy * scaleY,
        width: box.width * scaleX,
        height: box.height * scaleY,
        angle: box.angle
      };
    }

    function applyThermalColormap(imageData) {
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const luminance = 0.2989 * r + 0.587 * g + 0.114 * b;
        const normalized = Math.min(1, Math.max(0, luminance / 255));
        const [cr, cg, cb] = paletteLut(normalized);
        data[i] = cr;
        data[i + 1] = cg;
        data[i + 2] = cb;
      }
      return imageData;
    }

    function paletteLut(t) {
      if (t < 0.2) {
        return [36, 82 + t * 170, 255];
      }
      if (t < 0.4) {
        return [36, 255, 140 + (t - 0.2) * 575];
      }
      if (t < 0.65) {
        return [120 + (t - 0.4) * 540, 255, 24];
      }
      if (t < 0.85) {
        return [255, 200 + (t - 0.65) * 275, 60];
      }
      return [255, 80 + (t - 0.85) * 300, 24 + (t - 0.85) * 80];
    }

    function updateHudData(boxInfo) {
      const areaPct = boxInfo ? Math.round((boxInfo.width * boxInfo.height) / (VIDEO_WIDTH * VIDEO_HEIGHT) * 100) : 0;
      const temp = boxInfo ? 32 + Math.round(Math.min(28, areaPct * 0.45 + motionScalar * 30)) : 24;
      hudJson = {
        seed: seedHex,
        area: `${String(areaPct).padStart(2, '0')}%`,
        tmax: `${temp}°C`,
        state: boxInfo ? 'ACTIVE' : 'IDLE',
        view: boxInfo ? 'HAND' : 'WAIT',
        dot: boxInfo ? '#4cff96' : '#7f8fa4'
      };
      fakeSocketUpdate(JSON.parse(JSON.stringify(hudJson)));
    }

    function fakeSocketUpdate(payload) {
      hudJson = payload;
    }

    function drawHudOverlay() {
      hudLeft.textContent = `HAND.SYS_08 - CALOR\nSEED: ${hudJson.seed}`;
      hudRight.textContent = `AREA: ${hudJson.area}\nTMAX: ${hudJson.tmax}`;
      hudState.innerHTML = `<span class="status-dot" style="background:${hudJson.dot};"></span>STATE: ${hudJson.state}`;
      hudView.textContent = `VIEW: ${hudJson.view}`;
    }

    function updateHudOverlay() {
      drawHudOverlay();
    }

    function generateSeed() {
      return Math.floor(Math.random() * 0xffffff).toString(16).toUpperCase().padStart(6, '0');
    }

    if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
      toggleBtn.disabled = false;
    } else {
      toggleBtn.disabled = true;
      boot.querySelector('p').textContent = 'Webcam access is required. Please open this page in a secure browser that supports getUserMedia().';
    }

    updateHudOverlay();
  