import os
import uuid
import shutil
import subprocess
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import time
import re

app = FastAPI(title="ClipViral API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

jobs = {}

MIN_DURATION = 61
MAX_DURATION = 180

# ─────────────────────────────────────────────────────────────
# MOTS VIRAUX — détection sémantique
# ─────────────────────────────────────────────────────────────
VIRAL_WORDS = {
    # Choc / surprise
    "incroyable": 1.0, "choquant": 1.0, "unbelievable": 1.0, "shocking": 1.0,
    "jamais vu": 1.0, "never seen": 1.0, "impossible": 0.9, "insane": 1.0,
    "fou": 0.8, "crazy": 0.8, "wtf": 1.0, "omg": 1.0,
    # Émotion forte
    "incroyablement": 0.9, "tellement": 0.7, "vraiment": 0.6,
    "literally": 0.7, "actually": 0.6, "honestly": 0.7,
    # Révélation / secret
    "secret": 0.9, "révélation": 1.0, "vérité": 0.8, "truth": 0.8,
    "personne ne sait": 1.0, "nobody knows": 1.0, "finally": 0.8,
    # Humour / réaction
    "lol": 0.7, "haha": 0.7, "mort de rire": 0.9, "hilarant": 0.9,
    # Urgence / attention
    "attention": 0.8, "important": 0.7, "écoute": 0.7, "listen": 0.7,
    "stop": 0.7, "wait": 0.8, "attends": 0.8,
    # Superlatifs
    "meilleur": 0.8, "pire": 0.8, "best": 0.8, "worst": 0.8,
    "plus grand": 0.8, "biggest": 0.8, "premier": 0.7, "first ever": 1.0,
}

# ─────────────────────────────────────────────────────────────
# 1. ÉNERGIE AUDIO
# ─────────────────────────────────────────────────────────────
def extract_audio_energy(video_path: str, segment_duration: float = 2.0):
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])

    cmd_audio = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "22050",
        "-f", "f32le", "-"
    ]
    result_audio = subprocess.run(cmd_audio, capture_output=True)

    if len(result_audio.stdout) < 100:
        return np.linspace(0, duration, 100), np.ones(100) * 0.5, np.ones(100) * 0.5, duration

    samples = np.frombuffer(result_audio.stdout, dtype=np.float32)
    sr = 22050
    hop = int(segment_duration * sr)

    times, energies, emotions = [], [], []

    for i in range(0, len(samples) - hop, hop):
        chunk = samples[i:i+hop]
        rms = np.sqrt(np.mean(chunk**2))
        energies.append(float(rms))
        times.append(i / sr)

        # Détection émotionnelle : variance + zero-crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(chunk)))) / 2
        variance = np.var(chunk)
        emotion_score = float(np.clip(zcr * 3 + variance * 10, 0, 1))
        emotions.append(emotion_score)

    return np.array(times), np.array(energies), np.array(emotions), duration


# ─────────────────────────────────────────────────────────────
# 2. CHANGEMENTS DE SCÈNE
# ─────────────────────────────────────────────────────────────
def extract_scene_changes(video_path: str) -> dict:
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", "select='gt(scene,0.3)',metadata=print:file=-",
        "-an", "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    scene_scores = {}
    current_time = None

    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            try:
                current_time = float(line.split("pts_time:")[1].split()[0])
            except:
                pass
        if "lavfi.scene_score=" in line and current_time is not None:
            try:
                score = float(line.split("lavfi.scene_score=")[1].strip())
                scene_scores[current_time] = score
            except:
                pass

    return scene_scores


# ─────────────────────────────────────────────────────────────
# 3. WHISPER — transcription + score sémantique
# ─────────────────────────────────────────────────────────────
def transcribe_with_whisper(video_path: str) -> list:
    """
    Transcrit la vidéo avec Whisper (tiny = rapide, gratuit).
    Retourne une liste de segments [{start, end, text, viral_score}]
    """
    try:
        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(str(video_path), word_timestamps=True, language=None)

        segments = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").lower().strip()
            score = compute_semantic_score(text)
            segments.append({
                "start": float(seg["start"]),
                "end":   float(seg["end"]),
                "text":  seg.get("text", "").strip(),
                "viral_score": score,
            })
        return segments
    except Exception as e:
        print(f"Whisper error: {e}")
        return []


def compute_semantic_score(text: str) -> float:
    """Score viral basé sur les mots clés émotionnels."""
    text = text.lower()
    score = 0.0
    for word, weight in VIRAL_WORDS.items():
        if word in text:
            score += weight
    # Ponctuation forte = émotion
    score += text.count("!") * 0.3
    score += text.count("?") * 0.15
    return min(score, 1.0)


def whisper_to_timeline(segments: list, times: np.ndarray) -> np.ndarray:
    """Mappe les scores Whisper sur la timeline audio."""
    whisper_scores = np.zeros(len(times))
    for seg in segments:
        mask = (times >= seg["start"]) & (times <= seg["end"])
        whisper_scores[mask] = max(whisper_scores[mask].max(), seg["viral_score"])
    return whisper_scores


# ─────────────────────────────────────────────────────────────
# 4. SCORE VIRAL COMBINÉ (40/30/20/10)
# ─────────────────────────────────────────────────────────────
def compute_viral_score(times, audio_energies, emotions, scene_scores, whisper_segments, duration):
    # Normalise audio
    norm_audio = audio_energies / audio_energies.max() if audio_energies.max() > 0 else np.ones_like(audio_energies) * 0.5

    # Normalise émotions
    norm_emotions = emotions / emotions.max() if emotions.max() > 0 else np.zeros_like(emotions)

    # Scènes → timeline
    scene_boost = np.zeros(len(times))
    for scene_time, scene_score in scene_scores.items():
        idx = np.argmin(np.abs(times - scene_time))
        if idx < len(scene_boost):
            scene_boost[idx] += scene_score
    if scene_boost.max() > 0:
        scene_boost = scene_boost / scene_boost.max()

    # Whisper → timeline
    whisper_scores = whisper_to_timeline(whisper_segments, times)

    # Score combiné : 40% Whisper + 30% audio + 20% scènes + 10% émotions
    if whisper_scores.max() > 0:
        combined = (
            whisper_scores * 0.40 +
            norm_audio     * 0.30 +
            scene_boost    * 0.20 +
            norm_emotions  * 0.10
        )
    else:
        # Fallback sans Whisper : 70% audio + 20% scènes + 10% émotions
        combined = (
            norm_audio   * 0.70 +
            scene_boost  * 0.20 +
            norm_emotions * 0.10
        )

    window   = 7
    smoothed = np.convolve(combined, np.ones(window)/window, mode='same')
    return smoothed


# ─────────────────────────────────────────────────────────────
# 5. DÉTECTION DES CLIPS
# ─────────────────────────────────────────────────────────────
def find_viral_clips(times, smoothed, duration):
    peak_threshold   = np.percentile(smoothed, 85)
    extend_threshold = np.percentile(smoothed, 60)

    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= peak_threshold:
            if smoothed[i] >= smoothed[i-1] and smoothed[i] >= smoothed[i+1]:
                peaks.append(i)

    peaks.sort(key=lambda i: smoothed[i], reverse=True)
    clips = []

    for peak_idx in peaks:
        peak_time = times[peak_idx]
        if any(abs(peak_time - s) < 120 for s, e in clips):
            continue

        left = peak_idx
        while left > 0:
            if times[peak_idx] - times[left - 1] > MAX_DURATION / 2:
                break
            if smoothed[left - 1] >= extend_threshold:
                left -= 1
            else:
                break

        right = peak_idx
        while right < len(times) - 1:
            clip_duration = times[right + 1] - times[left]
            if clip_duration > MAX_DURATION:
                break
            if smoothed[right + 1] >= extend_threshold:
                right += 1
            else:
                if times[right] - times[left] < MIN_DURATION:
                    right += 1
                else:
                    break

        start_t  = float(times[left])
        end_t    = float(times[right])
        dur      = end_t - start_t

        if dur < MIN_DURATION:
            end_t = min(start_t + MIN_DURATION, duration)
        if end_t > duration:
            end_t = duration
            start_t = max(0, end_t - MIN_DURATION)

        mask = (times >= start_t) & (times <= end_t)
        viral_score = float(smoothed[mask].mean()) if mask.any() else 0.5

        clips.append({
            "start": round(start_t, 1),
            "end":   round(end_t, 1),
            "duration": round(end_t - start_t, 1),
            "viral_score": round(viral_score, 3),
        })

    clips.sort(key=lambda x: x["start"])
    return clips


# ─────────────────────────────────────────────────────────────
# 6. GÉNÉRATION CAPTION TIKTOK
# ─────────────────────────────────────────────────────────────
def generate_tiktok_caption(clip: dict, whisper_segments: list) -> str:
    """Génère une caption TikTok accrocheuse basée sur le contenu du clip."""
    # Trouver les segments Whisper dans la fenêtre du clip
    clip_texts = [
        s["text"] for s in whisper_segments
        if s["start"] >= clip["start"] and s["end"] <= clip["end"]
    ]

    # Trouver les mots viraux présents
    found_viral = []
    for text in clip_texts:
        for word in VIRAL_WORDS:
            if word in text.lower() and word not in found_viral:
                found_viral.append(word)

    score_pct = int(clip["viral_score"] * 100)

    # Templates de captions selon le score
    if score_pct >= 80:
        templates = [
            "POV : tu tombes sur le moment le plus fou 🔥",
            "Ce moment va te laisser sans voix 😱",
            "Ils ont pas coupé ça au montage... 👀",
            "Le moment que tout le monde attendait 💥",
        ]
    elif score_pct >= 60:
        templates = [
            "Ce passage mérite vraiment d'être vu 👇",
            "Le meilleur moment de la vidéo 🎯",
            "Regarde jusqu'à la fin, ça vaut le coup 🔥",
            "Ce moment-là, on en parle ? 👀",
        ]
    else:
        templates = [
            "Moment clé à ne pas rater 👇",
            "À voir absolument 🎬",
            "Ce passage change tout 💡",
        ]

    import random
    caption = random.choice(templates)

    # Ajoute hashtags automatiques
    hashtags = "#viral #tiktok #fyp #pourtoi"
    if found_viral:
        hashtags += " #" + found_viral[0].replace(" ", "")

    return f"{caption}\n\n{hashtags}"


# ─────────────────────────────────────────────────────────────
# 7. EXPORT CLIP AVEC SOUS-TITRES
# ─────────────────────────────────────────────────────────────
def create_srt(whisper_segments: list, start: float, end: float, out_path: str):
    """Crée un fichier SRT pour les sous-titres du clip."""
    srt_lines = []
    idx = 1
    for seg in whisper_segments:
        if seg["end"] < start or seg["start"] > end:
            continue
        seg_start = max(seg["start"] - start, 0)
        seg_end   = min(seg["end"] - start, end - start)

        def fmt(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            ms = int((s % 1) * 1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        srt_lines.append(f"{idx}\n{fmt(seg_start)} --> {fmt(seg_end)}\n{seg['text'].strip()}\n")
        idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    return len(srt_lines) > 0


def export_clip_tiktok(input_path: str, start: float, end: float, out_path: str,
                        srt_path: str = None):
    duration   = end - start
    fade_start = max(0, duration - 2.0)

    # Sous-titres brûlés dans la vidéo si SRT disponible
    subtitle_filter = ""
    if srt_path and Path(srt_path).exists() and Path(srt_path).stat().st_size > 10:
        # Escape le chemin pour FFmpeg
        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
        subtitle_filter = (
            f",subtitles={srt_escaped}:force_style='"
            f"FontName=Arial,FontSize=14,Bold=1,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"BackColour=&H80000000,Outline=2,Shadow=1,"
            f"Alignment=2,MarginV=80'"
        )

    vf = (
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,boxblur=20:20,eq=brightness=-0.4[bg];"
        f"[0:v]scale=1080:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        f"{subtitle_filter},"
        f"fade=t=out:st={fade_start}:d=2[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
        "-filter_complex", vf,
        "-map", "[out]",
        "-map", "0:a",
        "-af", f"afade=t=out:st={fade_start}:d=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────
# 8. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────
def process_video_job(job_id: str, video_path: str):
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        # ÉTAPE 1 — Énergie audio + émotions
        jobs[job_id]["status"]   = "analyzing"
        jobs[job_id]["progress"] = 8
        jobs[job_id]["message"]  = "🎵 Analyse de l'énergie audio..."

        times, energies, emotions, duration = extract_audio_energy(video_path)

        # ÉTAPE 2 — Changements de scène
        jobs[job_id]["progress"] = 20
        jobs[job_id]["message"]  = "🎬 Détection des changements de scène..."

        scene_scores = extract_scene_changes(video_path)

        # ÉTAPE 3 — Whisper transcription
        jobs[job_id]["progress"] = 32
        jobs[job_id]["message"]  = "🧠 Transcription Whisper AI en cours..."

        whisper_segments = transcribe_with_whisper(video_path)

        jobs[job_id]["progress"] = 50
        jobs[job_id]["message"]  = f"✍️ {len(whisper_segments)} segments transcrits. Calcul du score viral..."

        # ÉTAPE 4 — Score viral combiné
        smoothed = compute_viral_score(times, energies, emotions, scene_scores, whisper_segments, duration)
        clips    = find_viral_clips(times, smoothed, duration)

        if not clips:
            jobs[job_id]["status"]  = "error"
            jobs[job_id]["message"] = "Aucun clip viral détecté."
            return

        jobs[job_id]["clips_meta"] = clips
        jobs[job_id]["progress"]   = 55
        jobs[job_id]["message"]    = f"✅ {len(clips)} clips détectés. Export TikTok..."
        jobs[job_id]["status"]     = "exporting"

        exported = []
        for idx, clip in enumerate(clips):
            out_path = job_dir / f"Clip_Elite_{idx+1}.mp4"
            srt_path = job_dir / f"Clip_Elite_{idx+1}.srt"

            # Génère les sous-titres
            has_subs = create_srt(whisper_segments, clip["start"], clip["end"], str(srt_path))

            # Génère la caption TikTok
            caption = generate_tiktok_caption(clip, whisper_segments)

            # Export avec sous-titres
            success = export_clip_tiktok(
                video_path, clip["start"], clip["end"], str(out_path),
                srt_path=str(srt_path) if has_subs else None
            )

            if success:
                exported.append({
                    **clip,
                    "filename": f"Clip_Elite_{idx+1}.mp4",
                    "url":      f"/outputs/{job_id}/Clip_Elite_{idx+1}.mp4",
                    "rank":     idx + 1,
                    "caption":  caption,
                    "has_subtitles": has_subs,
                })

            progress = 55 + int(((idx + 1) / len(clips)) * 42)
            jobs[job_id]["progress"] = progress
            jobs[job_id]["message"]  = f"Export clip {idx+1}/{len(clips)}..."

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["clips"]    = exported
        jobs[job_id]["message"]  = f"🎉 {len(exported)} clips prêts !"

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)
    finally:
        try:
            os.remove(video_path)
        except:
            pass


# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Fichier vidéo requis.")

    job_id     = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status":     "queued",
        "progress":   0,
        "message":    "En attente...",
        "clips":      [],
        "created_at": time.time(),
    }

    background_tasks.add_task(process_video_job, job_id, str(video_path))
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job introuvable.")
    return jobs[job_id]


@app.get("/outputs/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, "Clip introuvable.")
    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# Serve frontend
import shutil as _shutil
static_path = Path("static")
if static_path.exists() and not static_path.is_dir():
    static_path.unlink()
static_path.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
