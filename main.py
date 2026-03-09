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
import httpx
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
MAX_VIDEO_DURATION = 3600  # 60 min max

# Clé API Claude — à définir en variable d'env sur Render
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─────────────────────────────────────────────────────────────
# CLEANUP AUTO — supprime les clips après 3h
# ─────────────────────────────────────────────────────────────
async def cleanup_loop():
    while True:
        await asyncio.sleep(600)  # vérifie toutes les 10 min
        now = time.time()
        for job_id, job in list(jobs.items()):
            age = now - job.get("created_at", now)
            if age > 10800:  # 3h
                job_dir = OUTPUT_DIR / job_id
                if job_dir.exists():
                    for f in job_dir.iterdir():
                        try: f.unlink()
                        except: pass
                    try: job_dir.rmdir()
                    except: pass
                jobs.pop(job_id, None)

@app.on_event("startup")
async def startup():
    asyncio.create_task(cleanup_loop())

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
# 1. DURÉE VIDÉO
# ─────────────────────────────────────────────────────────────
def _get_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return float(r.stdout.strip())
    except:
        return 0.0

# ─────────────────────────────────────────────────────────────
# 2. ANALYSE AUDIO PAR CHUNKS (anti-RAM crash)
# ─────────────────────────────────────────────────────────────
def _extract_audio_energy(video_path: str, segment_duration: float = 5.0):
    """Analyse audio par chunks de 5min — max ~50MB RAM."""
    duration = _get_duration(video_path)
    sr = 16000
    hop = int(segment_duration * sr)
    chunk_secs = 300
    times, energies, emotions = [], [], []

    offset = 0.0
    while offset < duration:
        end = min(offset + chunk_secs, duration)
        cmd = ["ffmpeg", "-y", "-ss", str(offset), "-to", str(end),
               "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr),
               "-f", "f32le", "-"]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if len(r.stdout) >= 100:
            samples = np.frombuffer(r.stdout, dtype=np.float32)
            for i in range(0, len(samples) - hop, hop):
                chunk = samples[i:i+hop]
                t = offset + i / sr
                times.append(t)
                energies.append(float(np.sqrt(np.mean(chunk**2))))
                zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
                emotions.append(float(np.clip(zcr * 3 + float(np.var(chunk)) * 10, 0, 1)))
            del samples
        offset += chunk_secs

    if not times:
        n = max(10, int(duration / segment_duration))
        return np.linspace(0, duration, n), np.ones(n)*0.5, np.ones(n)*0.5, duration

    return np.array(times), np.array(energies), np.array(emotions), duration

# ─────────────────────────────────────────────────────────────
# 3. SCÈNES — désactivé (trop lent sur Render free)
# ─────────────────────────────────────────────────────────────
def _compress_for_scenes(video_path: str, job_id: str) -> str:
    return str(video_path)

def _extract_scenes(video_path: str) -> dict:
    return {}

# ─────────────────────────────────────────────────────────────
# 4. WHISPER — transcription complète par sampling adaptatif
# ─────────────────────────────────────────────────────────────
def _transcribe_full(video_path: str, duration: float, job_id: str) -> list:
    """
    Sampling adaptatif sur toute la vidéo :
    - ≤30min : 100% couverture (30s/30s)
    - 30-60min : 1 sample/60s
    - >60min : 1 sample/90s
    Résultat : transcription couvrant toute la vidéo, timestamps exacts.
    """
    try:
        import whisper
        model = whisper.load_model("tiny")
    except:
        return []

    SAMPLE_DUR = 30
    if duration <= 1800:
        interval = 30
    elif duration <= 3600:
        interval = 60
    else:
        interval = 90

    sample_starts = []
    t = 0.0
    while t + SAMPLE_DUR <= duration:
        sample_starts.append(t)
        t += interval
    last = max(0, duration - SAMPLE_DUR)
    if not sample_starts or sample_starts[-1] < last - 5:
        sample_starts.append(last)

    total = len(sample_starts)
    all_segs = []

    for i, zs in enumerate(sample_starts):
        ze = min(zs + SAMPLE_DUR, duration)
        pct = 30 + int((i / total) * 30)
        set_job(job_id,
                progress=pct,
                message=f"🎙️ Transcription {i+1}/{total} · {int(zs//60)}:{int(zs%60):02d}...")

        tmp = str(UPLOAD_DIR / f"z_{job_id}_{i}.wav")
        cmd = ["ffmpeg", "-y", "-ss", str(zs), "-to", str(ze),
               "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0 or not Path(tmp).exists():
            continue
        try:
            res = model.transcribe(tmp, language=None, fp16=False, verbose=False,
                                   condition_on_previous_text=False)
            for seg in res.get("segments", []):
                text = seg.get("text", "").strip()
                if text:
                    all_segs.append({
                        "start": round(float(seg["start"]) + zs, 1),
                        "end":   round(float(seg["end"])   + zs, 1),
                        "text":  text,
                    })
        except Exception as e:
            print(f"Whisper sample {i} erreur: {e}")
        finally:
            try: os.remove(tmp)
            except: pass

    return all_segs

# ─────────────────────────────────────────────────────────────
# 5. CLAUDE API — scoring viral intelligent
# ─────────────────────────────────────────────────────────────
async def _claude_score_clips(segments: list, duration: float, job_id: str) -> list:
    """
    Envoie la transcription complète à Claude qui identifie
    les meilleurs moments viraux avec timestamps précis.
    Retourne une liste de clips triés par score décroissant.
    """
    if not ANTHROPIC_API_KEY:
        return []

    # Construit le texte de transcription
    transcript = "\n".join(
        f"[{int(s['start']//60)}:{int(s['start']%60):02d}] {s['text']}"
        for s in segments
    ) if segments else "Pas de transcription disponible."

    dur_min = int(duration // 60)
    dur_sec = int(duration % 60)

    prompt = f"""Tu es un expert en création de contenu viral TikTok/Reels.
Voici la transcription d'une vidéo de {dur_min}min{dur_sec}s.

TRANSCRIPTION :
{transcript[:12000]}

MISSION : Identifie les 5 meilleurs moments pour créer des clips viraux TikTok.
Chaque clip doit durer entre 61 et 180 secondes.

Critères de sélection (par ordre d'importance) :
1. Moments émotionnellement forts (rires, surprises, révélations, chocs)
2. Accroches puissantes (début de phrase qui donne envie de continuer)
3. Valeur informationnelle ou divertissante exceptionnelle
4. Moments avec des réactions authentiques
5. Punchlines ou citations mémorables

Réponds UNIQUEMENT avec ce JSON valide, rien d'autre :
{{
  "clips": [
    {{
      "start": 45.0,
      "end": 165.0,
      "score": 0.95,
      "reason": "Révélation choquante avec réaction authentique",
      "caption": "Ils ont caché ça pendant des années... 😱\\n\\n#viral #fyp #choquant #pourtoi"
    }}
  ]
}}

Règles :
- start et end sont en secondes (floats)
- score entre 0.0 et 1.0
- Minimum 120 secondes entre le début de chaque clip
- Ne dépasse pas {int(duration)}s pour end
- caption en français avec emojis et hashtags TikTok"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
        data = r.json()
        text = data["content"][0]["text"].strip()
        # Nettoie le JSON si Claude ajoute des backticks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        clips = result.get("clips", [])
        # Validation basique
        valid = []
        for c in clips:
            s = float(c.get("start", 0))
            e = float(c.get("end", 0))
            if e - s >= MIN_DURATION and e <= duration + 5:
                e = min(e, duration)
                valid.append({
                    "start":       round(s, 1),
                    "end":         round(e, 1),
                    "duration":    round(e - s, 1),
                    "viral_score": float(c.get("score", 0.7)),
                    "reason":      c.get("reason", ""),
                    "caption":     c.get("caption", ""),
                })
        return sorted(valid, key=lambda x: x["viral_score"], reverse=True)
    except Exception as e:
        print(f"Claude API erreur: {e}")
        return []

# ─────────────────────────────────────────────────────────────
# 6. FALLBACK — score audio si Claude API indisponible
# ─────────────────────────────────────────────────────────────
def _audio_fallback_clips(times, energies, emotions, duration) -> list:
    """Score 100% audio RMS — fallback si pas de clé API."""
    norm_a = energies / energies.max() if energies.max() > 0 else np.ones_like(energies)
    norm_e = emotions / emotions.max() if emotions.max() > 0 else np.zeros_like(emotions)
    combined = norm_a * 0.70 + norm_e * 0.30
    w = max(5, min(20, int(len(times) / 40)))
    smoothed = np.convolve(combined, np.ones(w)/w, mode='same')

    sr = np.percentile(smoothed, 75)
    candidates = np.where(smoothed >= sr)[0]
    clips = []
    last_end = -MIN_GAP

    for idx in candidates:
        t = float(times[idx])
        if t - last_end < MIN_GAP:
            continue
        left, right = idx, idx
        sr2 = len(times)
        while right < sr2 - 1 and times[right] - times[left] < MIN_DURATION:
            right += 1
        s = float(times[left])
        e = min(float(times[right]), duration)
        if e - s < MIN_DURATION:
            e = min(s + MIN_DURATION, duration)
        if e - s > MAX_DURATION:
            e = s + MAX_DURATION
        mask = (times >= s) & (times <= e)
        clips.append({
            "start":       round(s, 1),
            "end":         round(e, 1),
            "duration":    round(e - s, 1),
            "viral_score": round(float(smoothed[mask].mean()) if mask.any() else 0.5, 3),
            "reason":      "Pic d'énergie audio",
            "caption":     "",
        })
        last_end = e
        if len(clips) >= 5:
            break

    return sorted(clips, key=lambda x: x["viral_score"], reverse=True)

# ─────────────────────────────────────────────────────────────
# 7. GÉNÉRATION CAPTION FALLBACK
# ─────────────────────────────────────────────────────────────
def _make_caption(clip) -> str:
    if clip.get("caption"):
        return clip["caption"]
    pct = int(clip.get("viral_score", 0.5) * 100)
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
    return f"{t}\n\n#viral #tiktok #fyp #pourtoi"

# ─────────────────────────────────────────────────────────────
# 8. EXPORT TIKTOK 1080×1920
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

# ─────────────────────────────────────────────────────────────
# 9. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────
async def process_video_job(job_id: str, video_path: str, filename: str):
    job_dir    = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    loop       = asyncio.get_event_loop()

    async def upd(progress, message, **kw):
        set_job(job_id, progress=progress, message=message, **kw)
        await notify_ws(job_id)

    try:
        # ÉTAPE 1 — Vérification durée
        size_mb = int(Path(video_path).stat().st_size / 1024 / 1024)
        title = Path(filename).stem[:60]
        set_job(job_id, video_title=title)
        await upd(5, f"✅ Vidéo reçue ({size_mb} MB). Vérification...")

        duration = await loop.run_in_executor(executor, _get_duration, video_path)
        if duration > MAX_VIDEO_DURATION:
            raise Exception(
    f"Vidéo trop longue ({int(duration//60)}min {int(duration%60)}s). "
    f"Maximum 60 min sur ce serveur. "
    f"Découpe la vidéo en morceaux de 60min avec VLC (gratuit) puis envoie chaque morceau séparément."
)

        mins = int(duration // 60)
        secs = int(duration % 60)
        await upd(10, f"⏱️ Durée : {mins}min {secs}s. Analyse audio...")

        # ÉTAPE 2 — Audio RMS (toute la vidéo, par chunks)
        times, energies, emotions, duration = await loop.run_in_executor(
            executor, _extract_audio_energy, video_path, 5.0)
        await upd(25, f"🎵 Audio analysé sur {mins}min. Transcription Whisper...")

        # ÉTAPE 3 — Whisper sampling adaptatif
        segments = await loop.run_in_executor(
            executor, _transcribe_full, video_path, duration, job_id)
        seg_count = len(segments)
        await upd(60, f"✍️ {seg_count} segments transcrits. Analyse IA...")

        # ÉTAPE 4 — Claude API scoring (meilleur) ou fallback audio
        if ANTHROPIC_API_KEY:
            await upd(62, "🧠 Claude analyse les meilleurs moments...")
            clips = await _claude_score_clips(segments, duration, job_id)
            method = "Claude AI"
        else:
            clips = []
            method = "Audio RMS"

        # Fallback si Claude renvoie rien ou pas de clé
        if not clips:
            await upd(65, f"⚡ Score audio ({method})...")
            clips = await loop.run_in_executor(
                executor, _audio_fallback_clips, times, energies, emotions, duration)

        if not clips:
            raise Exception("Aucun passage hype détecté. Essaie avec une vidéo plus longue.")

        await upd(68, f"✅ {len(clips)} clips détectés via {method}. Export TikTok...", status="exporting")

        # ÉTAPE 5 — Export TikTok
        exported = []
        for idx, clip in enumerate(clips[:5]):  # max 5 clips
            out_path = job_dir / f"Clip_Elite_{idx+1}.mp4"
            caption  = _make_caption(clip)
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
                    "reason":      clip.get("reason", ""),
                })
            pct = 68 + int(((idx+1) / min(len(clips), 5)) * 30)
            await upd(pct, f"🎬 Export {idx+1}/{min(len(clips), 5)} ({int(clip['duration'])}s)...")

        set_job(job_id, status="done", progress=100, clips=exported,
                message=f"🎉 {len(exported)} clips viraux prêts !")
        await notify_ws(job_id)

    except Exception as e:
        set_job(job_id, status="error", message=f"❌ {str(e)}")
        await notify_ws(job_id)
    finally:
        try:
            if Path(video_path).exists():
                os.remove(video_path)
        except: pass

# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, request: Request):
    job_id   = str(uuid.uuid4())[:8]
    out_path = str(UPLOAD_DIR / f"{job_id}.mp4")

    content_disp = request.headers.get("content-disposition", "")
    filename = "video.mp4"
    if "filename=" in content_disp:
        try: filename = content_disp.split("filename=")[1].strip().strip('"')
        except: pass

    size = 0
    with open(out_path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
            if size > 4 * 1024 * 1024 * 1024:
                os.remove(out_path)
                raise HTTPException(413, "Fichier trop volumineux (max 4GB).")

    if size < 10000:
        if Path(out_path).exists(): os.remove(out_path)
        raise HTTPException(400, "Fichier invalide ou vide.")

    jobs[job_id] = {
        "status": "queued", "progress": 2,
        "message": "⬆️ Fichier reçu, initialisation...",
        "clips": [], "created_at": time.time(), "video_title": "",
    }
    background_tasks.add_task(process_video_job, job_id, out_path, filename)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job introuvable.")
    return jobs[job_id]

@app.get("/api/jobs")
async def list_jobs():
    """Retourne tous les jobs terminés pour le panneau fichiers."""
    result = []
    for jid, job in jobs.items():
        if job.get("status") == "done" and job.get("clips"):
            result.append({
                "job_id":      jid,
                "video_title": job.get("video_title", "Vidéo"),
                "clips":       job.get("clips", []),
                "created_at":  job.get("created_at", 0),
            })
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result

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
