
(function () {
  "use strict";

  const DIGITS = ["۰","۱","۲","۳","۴","۵","۶","۷","۸","۹"];

  window.classifyFarsiDigit = function (
    imageData, sourceWidth, sourceHeight, box
  ) {
    const weights = window.MNIST_WEIGHTS;
    if (!weights || !weights.weightsHidden ||
        !weights.weightsOutput) {
      throw new Error("مدل بارگذاری نشده است.");
    }

    const data = imageData.data;
    const x0 = Math.max(0, Math.floor(box.x));
    const y0 = Math.max(0, Math.floor(box.y));
    const x1 = Math.min(sourceWidth, Math.ceil(box.x + box.w));
    const y1 = Math.min(sourceHeight, Math.ceil(box.y + box.h));

    let left = sourceWidth, top = sourceHeight;
    let right = -1, bottom = -1;

    function isRed(x, y) {
      const p = (y * sourceWidth + x) * 4;
      const r = data[p], g = data[p + 1], b = data[p + 2];
      return r > 100 && r > g * 1.35 &&
             r > b * 1.25 && g < 170;
    }

    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        if (isRed(x, y)) {
          left = Math.min(left, x);
          right = Math.max(right, x);
          top = Math.min(top, y);
          bottom = Math.max(bottom, y);
        }
      }
    }

    if (right < left || bottom < top) return null;

    const bw = right - left + 1;
    const bh = bottom - top + 1;
    const scale = Math.min(20 / bw, 20 / bh);
    const dw = Math.max(1, Math.round(bw * scale));
    const dh = Math.max(1, Math.round(bh * scale));
    const ox = Math.floor((28 - dw) / 2);
    const oy = Math.floor((28 - dh) / 2);
    const pixels = new Float32Array(784);

    for (let dy = 0; dy < dh; dy++) {
      for (let dx = 0; dx < dw; dx++) {
        const sx = Math.min(
          right, left + Math.floor(dx / scale)
        );
        const sy = Math.min(
          bottom, top + Math.floor(dy / scale)
        );
        if (isRed(sx, sy)) {
          pixels[(oy + dy) * 28 + ox + dx] = 1;
        }
      }
    }

    const pooled = new Float32Array(197);
    for (let y = 0; y < 14; y++) {
      for (let x = 0; x < 14; x++) {
        const i = y * 2 * 28 + x * 2;
        pooled[y * 14 + x] =
          (pixels[i] + pixels[i + 1] +
           pixels[i + 28] + pixels[i + 29]) / 4;
      }
    }
    pooled[196] = 1;

    const hidden = new Float32Array(101);
    for (let i = 0; i < 100; i++) {
      let sum = 0;
      for (let j = 0; j < 197; j++) {
        sum += pooled[j] * weights.weightsHidden[i * 197 + j];
      }
      hidden[i] = Math.max(0, sum);
    }
    hidden[100] = 1;

    const scores = new Float64Array(10);
    let max = -Infinity;
    for (let i = 0; i < 10; i++) {
      let sum = 0;
      for (let j = 0; j < 101; j++) {
        sum += hidden[j] * weights.weightsOutput[i * 101 + j];
      }
      scores[i] = sum;
      max = Math.max(max, sum);
    }

    let total = 0;
    for (let i = 0; i < 10; i++) {
      scores[i] = Math.exp(scores[i] - max);
      total += scores[i];
    }

    let best = 0;
    for (let i = 0; i < 10; i++) {
      scores[i] /= total;
      if (scores[i] > scores[best]) best = i;
    }

    return {
      digit: DIGITS[best],
      confidence: scores[best],
      index: best
    };
  };
})();
