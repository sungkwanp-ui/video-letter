// 받아쓰기 워커: 브라우저 안에서 Whisper 모델을 돌린다 (서버로 녹음이 전송되지 않음).
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

const SR = 16000;
let asr = null;
let loaded = null; // { model, device }

const post = (m) => self.postMessage(m);

async function load(model, device) {
  if (loaded && loaded.model === model && loaded.device === device) return;
  asr = null;
  const dtype = device === 'webgpu'
    ? { encoder_model: 'fp32', decoder_model_merged: 'q4' }
    : 'q8';
  const seen = {};
  asr = await pipeline('automatic-speech-recognition', model, {
    device,
    dtype,
    progress_callback: (p) => {
      if (p.status === 'progress' && p.total) {
        seen[p.file] = [p.loaded, p.total];
        let l = 0, t = 0;
        for (const k in seen) { l += seen[k][0]; t += seen[k][1]; }
        post({ type: 'status', text: `음성 인식 모델 내려받는 중… ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB (처음 한 번만)` });
      }
    },
  });
  loaded = { model, device };
}

// 조용한 지점을 찾아 30초 이하 조각으로 자른다.
function splitPoints(audio) {
  const hop = SR / 10; // 0.1초
  const n = Math.floor(audio.length / hop);
  const rms = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i * hop; j < (i + 1) * hop; j++) s += audio[j] * audio[j];
    rms[i] = Math.sqrt(s / hop);
  }
  const pieces = [];
  let pos = 0;
  while (pos < audio.length) {
    let end;
    if (audio.length - pos <= 29 * SR) {
      end = audio.length;
    } else {
      const a = Math.floor((pos + 18 * SR) / hop), b = Math.floor((pos + 29 * SR) / hop);
      let best = a;
      for (let i = a; i < b && i < n; i++) if (rms[i] < rms[best]) best = i;
      end = Math.floor((best + 0.5) * hop);
    }
    let s = 0;
    for (let j = pos; j < end; j += 8) s += audio[j] * audio[j];
    pieces.push({ start: pos, end, level: Math.sqrt(s / ((end - pos) / 8)) });
    pos = end;
  }
  return pieces;
}

async function transcribe(audio, model, language, device) {
  await load(model, device);
  post({ type: 'status', text: `받아쓰는 중… (${device === 'webgpu' ? '그래픽 가속' : '일반 모드'})` });
  const pieces = splitPoints(audio);
  const segs = [];
  for (let i = 0; i < pieces.length; i++) {
    const p = pieces[i];
    const off = p.start / SR, dur = (p.end - p.start) / SR;
    if (p.level > 0.003 && dur > 0.3) {
      const out = await asr(audio.subarray(p.start, p.end), {
        language, task: 'transcribe', return_timestamps: true,
      });
      const chunks = out.chunks && out.chunks.length ? out.chunks : [{ timestamp: [0, dur], text: out.text }];
      for (const c of chunks) {
        const text = (c.text || '').trim();
        if (!text) continue;
        const a = Math.min(c.timestamp[0] ?? 0, dur);
        const b = Math.min(c.timestamp[1] ?? dur, dur);
        segs.push({ start: off + a, end: off + Math.max(b, a), text });
      }
    }
    post({ type: 'progress', done: i + 1, total: pieces.length });
  }
  return segs;
}

self.onmessage = async (e) => {
  const { audio, model, language } = e.data;
  const devices = [];
  try { if (self.navigator.gpu && await self.navigator.gpu.requestAdapter()) devices.push('webgpu'); } catch (_) {}
  devices.push('wasm');
  let lastErr;
  for (const d of devices) {
    try {
      const segs = await transcribe(audio, model, language, d);
      post({ type: 'done', segs, device: d });
      return;
    } catch (err) {
      lastErr = err;
      loaded = null;
      console.warn('받아쓰기 실패, 다음 방식으로 재시도:', d, err);
    }
  }
  post({ type: 'error', message: String(lastErr && lastErr.message || lastErr) });
};
