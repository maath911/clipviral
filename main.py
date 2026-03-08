import os
import uuid
import subprocess
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="ClipViral API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

jobs: dict = {}
ws_connections: dict[str, list] = {}
executor = ThreadPoolExecutor(max_workers=2)

MIN_DURATION = 61
MAX_DURATION = 180
MIN_GAP      = 120

VIRAL_WORDS = {
    "incroyable": 1.0, "choquant": 1.0, "unbelievable": 1.0, "shocking": 1.0,
    "jamais vu": 1.0, "never seen": 1.0, "impossible": 0.9, "insane": 1.0,
    "fou": 0.8, "crazy": 0.8, "wtf": 1.0, "omg": 1.0, "waouh": 1.0, "wow": 1.0,
    "incroyablement": 0.9, "literally": 0.7, "honestly": 0.7,
    "secret": 0.9, "révélation": 1.0, "vérité": 0.8, "truth": 0.8,
    "personne ne sait": 1.0, "nobody knows": 1.0, "finally": 0.8,
    "lol": 0.7, "haha": 0.7, "mort de rire": 0.9, "hilarant": 0.9,
    "attention": 0.8, "stop": 0.7, "wait": 0.8, "attends": 0.8,
    "meilleur": 0.8, "pire": 0.8, "best": 0.8, "worst": 0.8,
    "first ever": 1.0, "putain": 0.9, "incredible": 1.0, "amazing": 0.9,
    "never": 0.7, "jamais": 0.7, "regarde": 0.7, "look": 0.7,
}

# ─────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────
async def notify_ws(job_id: str):
    if job_id not in ws_connections:
        return
    data = jobs.get(job_id, {})
    dead = []
    for ws in ws_connections[job_id]:
        try:
            await ws.send_json(data)
        except:
            dead.append(ws)
    for ws in dead:
        ws_connections[job_id].remove(ws)

def set_job(job_id: str, **kwargs):
    jobs[job_id].update(kwargs)

# ─────────────────────────────────────────────────────────────
# 1. TÉLÉCHARGEMENT YOUTUBE
# ─────────────────────────────────────────────────────────────
def _download_youtube(url: str, job_id: str) -> str:
    """Télécharge la vidéo YouTube avec yt-dlp."""
    out_path = str(UPLOAD_DIR / f"{job_id}.mp4")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-warnings",
        "-o", out_path,
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise Exception(f"Téléchargement échoué : {result.stderr[:200]}")
    if not Path(out_path).exists():
        raise Exception("Fichier téléchargé introuvable.")
    return out_path

def _get_video_title(url: str) -> str:
    """Récupère le titre de la vidéo YouTube."""
    try:
        cmd = ["yt-dlp", "--get-title", "--no-warnings", url]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()[:80] if r.returncode == 0 else ""
    except:
        return ""

# ─────────────────────────────────────────────────────────────
# 2. ANALYSE AUDIO
# ─────────────────────────────────────────────────────────────
def _get_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", str(video_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except:
        return 300.0

def _extract_audio_energy(video_path: str, segment_duration: float = 5.0):
    duration = _get_duration(video_path)
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if len(r.stdout) < 100:
        n = max(10, int(duration / segment_duration))
        return np.linspace(0, duration, n), np.ones(n)*0.5, np.ones(n)*0.5, duration

    samples = np.frombuffer(r.stdout, dtype=np.float32)
    sr, hop = 16000, int(segment_duration * 16000)
    times, energies, emotions = [], [], []
    for i in range(0, len(samples) - hop, hop):
        chunk = samples[i:i+hop]
        times.append(i / sr)
        energies.append(float(np.sqrt(np.mean(chunk**2))))
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
        emotions.append(float(np.clip(zcr * 3 + float(np.var(chunk)) * 10, 0, 1)))
    return np.array(times), np.array(energies), np.array(emotions), duration

# ─────────────────────────────────────────────────────────────
# 3. DÉTECTION SCÈNES
# ─────────────────────────────────────────────────────────────
def _compress_for_scenes(video_path: str, job_id: str) -> str:
    out = str(UPLOAD_DIR / f"sc_{job_id}.mp4")
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-vf", "scale=-2:360,fps=2",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35", "-an", out]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    return out if r.returncode == 0 and Path(out).exists() else str(video_path)

def _extract_scenes(video_path: str) -> dict:
    cmd = ["ffmpeg", "-i", str(video_path),
           "-vf", "select='gt(scene,0.25)',metadata=print:file=-",
           "-an", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    scores, t = {}, None
    for line in r.stderr.split("\n"):
        if "pts_time:" in line:
            try: t = float(line.split("pts_time:")[1].split()[0])
            except: pass
        if "lavfi.scene_score=" in line and t is not None:
            try: scores[t] = float(line.split("lavfi.scene_score=")[1].strip())
            except: pass
    return scores

# ─────────────────────────────────────────────────────────────
# 4. ZONES CHAUDES
# ─────────────────────────────────────────────────────────────
def _find_hot_zones(times, energies, emotions, scene_scores, duration, zone=210):
    norm_a = energies / energies.max() if energies.max() > 0 else np.ones_like(energies)
    norm_e = emotions / emotions.max() if emotions.max() > 0 else np.zeros_like(emotions)
    boost  = np.zeros(len(times))
    for st, sc in scene_scores.items():
        idx = np.argmin(np.abs(times - st))
        if idx < len(boost): boost[idx] += sc
    if boost.max() > 0: boost /= boost.max()

    combined = norm_a * 0.65 + boost * 0.25 + norm_e * 0.10
    w = max(5, min(20, int(len(times) / 40)))
    smoothed = np.convolve(combined, np.ones(w)/w, mode='same')

    blocks = []
    for b in range(max(1, int(duration / zone))):
        bs, be = b * zone, min((b+1)*zone, duration)
        mask = (times >= bs) & (times < be)
        if mask.any():
            blocks.append((bs, be, float(smoothed[mask].mean())))

    if not blocks:
        return [(0, min(zone, duration))], smoothed

    thr = np.percentile([b[2] for b in blocks], 40)
    hot = sorted([(s, e) for s, e, sc in blocks if sc >= thr], key=lambda x: x[0])
    return (hot if hot else [(0, min(zone, duration))]), smoothed

# ─────────────────────────────────────────────────────────────
# 5. WHISPER SUR ZONES CHAUDES
# ─────────────────────────────────────────────────────────────
def _transcribe_zones(video_path: str, hot_zones: list, job_id: str) -> list:
    try:
        import whisper
        model = whisper.load_model("tiny")
    except:
        return []

    all_segs = []
    for i, (zs, ze) in enumerate(hot_zones):
        set_job(job_id,
                progress=38 + int((i / len(hot_zones)) * 17),
                message=f"🧠 Whisper zone {i+1}/{len(hot_zones)} ({int(zs//60)}min → {int(ze//60)}min)...")

        tmp = str(UPLOAD_DIR / f"z_{job_id}_{i}.wav")
        cmd = ["ffmpeg", "-y", "-ss", str(zs), "-to", str(ze),
               "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or not Path(tmp).exists():
            continue
        try:
            res = model.transcribe(tmp, language=None, fp16=False, verbose=False)
            for seg in res.get("segments", []):
                text  = seg.get("text", "").strip()
                score = sum(w for k, w in VIRAL_WORDS.items() if k in text.lower())
                score += text.count("!")*0.3 + text.count("?")*0.15
                all_segs.append({
                    "start": float(seg["start"]) + zs,
                    "end":   float(seg["end"])   + zs,
                    "text":  text,
                    "viral_score": min(score, 1.0),
                })
        except Exception as e:
            print(f"Whisper zone {i} erreur: {e}")
        finally:
            try: os.remove(tmp)
            except: pass
    return all_segs

# ─────────────────────────────────────────────────────────────
# 6. SCORE VIRAL FINAL
# ─────────────────────────────────────────────────────────────
def _compute_score(times, energies, emotions, scene_scores, whisper_segs, duration):
    norm_a = energies / energies.max() if energies.max() > 0 else np.ones_like(energies)*0.5
    norm_e = emotions / emotions.max() if emotions.max() > 0 else np.zeros_like(emotions)
    boost  = np.zeros(len(times))
    for st, sc in scene_scores.items():
        idx = np.argmin(np.abs(times - st))
        if idx < len(boost): boost[idx] += sc
    if boost.max() > 0: boost /= boost.max()

    ws = np.zeros(len(times))
    for seg in whisper_segs:
        mask = (times >= seg["start"]) & (times <= seg["end"])
        if mask.any(): ws[mask] = max(ws[mask].max(), seg["viral_score"])

    combined = (ws*0.40 + norm_a*0.35 + boost*0.15 + norm_e*0.10) \
        if ws.max() > 0 else (norm_a*0.70 + boost*0.20 + norm_e*0.10)

    w = max(5, min(15, int(len(times)/50)))
    return np.convolve(combined, np.ones(w)/w, mode='same')

# ─────────────────────────────────────────────────────────────
# 7. DÉTECTION CLIPS (61s → 180s, tous les passages hype)
# ─────────────────────────────────────────────────────────────
def _find_clips(times, smoothed, duration):
    peak_thr   = np.percentile(smoothed, 80)
    extend_thr = np.percentile(smoothed, 55)
    peaks = [i for i in range(1, len(smoothed)-1)
             if smoothed[i] >= peak_thr
             and smoothed[i] >= smoothed[i-1]
             and smoothed[i] >= smoothed[i+1]]
    peaks.sort(key=lambda i: smoothed[i], reverse=True)
    clips = []
    for pi in peaks:
        pt = times[pi]
        if any(abs(pt - s) < MIN_GAP for s, e in clips):
            continue
        left = pi
        while left > 0:
            if times[pi] - times[left-1] > MAX_DURATION/2: break
            if smoothed[left-1] >= extend_thr: left -= 1
            else: break
        right = pi
        while right < len(times)-1:
            if times[right+1] - times[left] > MAX_DURATION: break
            if smoothed[right+1] >= extend_thr: right += 1
            else:
                if times[right] - times[left] < MIN_DURATION: right += 1
                else: break
        s = float(times[left])
        e = float(times[right])
        if e - s < MIN_DURATION: e = min(s + MIN_DURATION, duration)
        if e > duration: e = duration; s = max(0, e - MIN_DURATION)
        if e - s > MAX_DURATION: e = s + MAX_DURATION
        mask = (times >= s) & (times <= e)
        clips.append({
            "start": round(s,1), "end": round(e,1),
            "duration": round(e-s,1),
            "viral_score": round(float(smoothed[mask].mean()) if mask.any() else 0.5, 3),
        })
    clips.sort(key=lambda x: x["start"])
    return clips

# ─────────────────────────────────────────────────────────────
# 8. EXPORT TIKTOK
# ─────────────────────────────────────────────────────────────
def _export_clip(input_path, start, end, out_path) -> bool:
    dur  = end - start
    fade = max(0, dur - 2.0)
    vf = (
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,boxblur=20:20,eq=brightness=-0.4[bg];"
        f"[0:v]scale=1080:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fade=t=out:st={fade}:d=2[out]"
    )
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", str(input_path),
        "-t", str(dur), "-filter_complex", vf,
        "-map", "[out]", "-map", "0:a",
        "-af", f"afade=t=out:st={fade}:d=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    return r.returncode == 0

def _make_caption(clip, whisper_segs):
    texts = [s["text"] for s in whisper_segs
             if s["start"] >= clip["start"] and s["end"] <= clip["end"]]
    viral_found = []
    for t in texts:
        for w in VIRAL_WORDS:
            if w in t.lower() and w not in viral_found:
                viral_found.append(w)
    pct = int(clip["viral_score"] * 100)
    if pct >= 80:
        t = random.choice([
            "POV : tu tombes sur le moment le plus fou 🔥",
            "Ce moment va te laisser sans voix 😱",
            "Ils ont pas coupé ça au montage... 👀",
            "Le moment que tout le monde attendait 💥",
        ])
    elif pct >= 60:
        t = random.choice([
            "Ce passage mérite vraiment d'être vu 👇",
            "Le meilleur moment de la vidéo 🎯",
            "Regarde jusqu'à la fin 🔥",
        ])
    else:
        t = random.choice([
            "Moment clé à ne pas rater 👇",
            "À voir absolument 🎬",
        ])
    tags = "#viral #tiktok #fyp #pourtoi"
    if viral_found:
        tags += " #" + viral_found[0].replace(" ", "")
    return f"{t}\n\n{tags}"

# ─────────────────────────────────────────────────────────────
# 9. PIPELINE PRINCIPAL ASYNC
# ─────────────────────────────────────────────────────────────
async def process_url_job(job_id: str, url: str):
    job_dir    = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    video_path = None
    scene_path = None
    loop       = asyncio.get_event_loop()

    async def upd(progress, message, **kw):
        set_job(job_id, progress=progress, message=message, **kw)
        await notify_ws(job_id)

    try:
        # ÉTAPE 1 — Titre YouTube
        await upd(2, "🔍 Récupération des infos de la vidéo...")
        title = await loop.run_in_executor(executor, _get_video_title, url)
        if title:
            set_job(job_id, video_title=title)

        # ÉTAPE 2 — Téléchargement
        await upd(5, f"⬇️ Téléchargement en cours... {'«' + title + '»' if title else ''}")
        video_path = await loop.run_in_executor(executor, _download_youtube, url, job_id)
        size_mb = int(Path(video_path).stat().st_size / 1024 / 1024)
        await upd(18, f"✅ Vidéo téléchargée ({size_mb} MB). Analyse audio...")

        # ÉTAPE 3 — Audio
        times, energies, emotions, duration = await loop.run_in_executor(
            executor, _extract_audio_energy, video_path, 5.0)
        mins = int(duration // 60); secs = int(duration % 60)
        await upd(28, f"🎵 Audio analysé · {mins}min {secs}s. Détection des scènes...")

        # ÉTAPE 4 — Scènes
        scene_path = await loop.run_in_executor(executor, _compress_for_scenes, video_path, job_id)
        scene_scores = await loop.run_in_executor(executor, _extract_scenes, scene_path)
        await upd(35, "🔥 Identification des passages hype...")

        # ÉTAPE 5 — Zones chaudes
        hot_zones, _ = await loop.run_in_executor(
            executor, _find_hot_zones, times, energies, emotions, scene_scores, duration, 210)
        total_min = round(sum(e - s for s, e in hot_zones) / 60, 1)
        await upd(38, f"✅ {len(hot_zones)} zones hype · {total_min} min → Whisper AI...")

        # ÉTAPE 6 — Whisper
        whisper_segs = await loop.run_in_executor(
            executor, _transcribe_zones, video_path, hot_zones, job_id)
        await upd(56, f"✍️ {len(whisper_segs)} segments analysés. Score viral...")

        # ÉTAPE 7 — Score final
        smoothed = await loop.run_in_executor(
            executor, _compute_score, times, energies, emotions,
            scene_scores, whisper_segs, duration)
        clips = await loop.run_in_executor(executor, _find_clips, times, smoothed, duration)

        if not clips:
            await upd(100, "❌ Aucun passage hype détecté.", status="error")
            return

        await upd(60, f"✅ {len(clips)} passages hype ! Export TikTok 1080×1920...", status="exporting")

        # ÉTAPE 8 — Export
        exported = []
        for idx, clip in enumerate(clips):
            out_path = job_dir / f"Clip_Elite_{idx+1}.mp4"
            caption  = _make_caption(clip, whisper_segs)
            success  = await loop.run_in_executor(
                executor, _export_clip, video_path, clip["start"], clip["end"], str(out_path))
            if success:
                exported.append({
                    **clip,
                    "filename":    f"Clip_Elite_{idx+1}.mp4",
                    "url":         f"/outputs/{job_id}/Clip_Elite_{idx+1}.mp4",
                    "preview_url": f"/outputs/{job_id}/Clip_Elite_{idx+1}.mp4",
                    "rank":        idx + 1,
                    "caption":     caption,
                    "video_title": title,
                })
            pct = 60 + int(((idx+1) / len(clips)) * 38)
            await upd(pct, f"🎬 Export {idx+1}/{len(clips)} ({int(clip['duration'])}s)...")

        set_job(job_id, status="done", progress=100, clips=exported,
                message=f"🎉 {len(exported)} clips viraux prêts à poster !")
        await notify_ws(job_id)

    except Exception as e:
        set_job(job_id, status="error", message=f"❌ {str(e)}")
        await notify_ws(job_id)
    finally:
        for f in [video_path, scene_path]:
            if f:
                try: os.remove(f)
                except: pass

# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────
class URLRequest(BaseModel):
    url: str

@app.post("/api/analyze")
async def analyze_url(body: URLRequest, background_tasks: BackgroundTasks):
    url = body.url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "URL invalide.")
    if not any(d in url for d in ["youtube.com", "youtu.be", "youtube"]):
        raise HTTPException(400, "Seules les URLs YouTube sont supportées pour l'instant.")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued", "progress": 0,
        "message": "⏳ Initialisation...",
        "clips": [], "created_at": time.time(),
        "video_title": "",
    }
    background_tasks.add_task(process_url_job, job_id, url)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job introuvable.")
    return jobs[job_id]

@app.websocket("/ws/{job_id}")
async def ws_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    ws_connections.setdefault(job_id, []).append(websocket)
    if job_id in jobs:
        await websocket.send_json(jobs[job_id])
    try:
        while True:
            await asyncio.sleep(15)
            try: await websocket.send_json({"ping": True})
            except: break
            if jobs.get(job_id, {}).get("status") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in ws_connections and websocket in ws_connections[job_id]:
            ws_connections[job_id].remove(websocket)

@app.get("/outputs/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    fp = OUTPUT_DIR / job_id / filename
    if not fp.exists():
        raise HTTPException(404, "Clip introuvable.")
    return FileResponse(str(fp), media_type="video/mp4", filename=filename,
                        headers={"Accept-Ranges": "bytes",
                                 "Content-Disposition": f"inline; filename={filename}"})

static_path = Path("static")
if static_path.exists() and not static_path.is_dir():
    static_path.unlink()
static_path.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
