import os
import uuid
import shutil
import subprocess
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import time
import asyncio
import random

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
ws_connections: dict[str, list[WebSocket]] = {}

MIN_DURATION = 61    # secondes minimum par clip
MAX_DURATION = 180   # secondes maximum par clip
MIN_GAP      = 120   # gap minimum entre deux clips (évite les doublons)

# ─────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────
async def notify_job(job_id: str, data: dict):
    if job_id in ws_connections:
        dead = []
        for ws in ws_connections[job_id]:
            try:
                await ws.send_json(data)
            except:
                dead.append(ws)
        for ws in dead:
            ws_connections[job_id].remove(ws)

def update_job(job_id: str, **kwargs):
    jobs[job_id].update(kwargs)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(notify_job(job_id, jobs[job_id]))
    except:
        pass

# ─────────────────────────────────────────────────────────────
# MOTS VIRAUX (pour scoring sémantique Whisper)
# ─────────────────────────────────────────────────────────────
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
    "first ever": 1.0, "putain": 0.9, "merde": 0.8,
    "regarde": 0.7, "look": 0.7, "incredible": 1.0, "amazing": 0.9,
    "never": 0.7, "jamais": 0.7, "toujours": 0.6, "always": 0.6,
}

# ─────────────────────────────────────────────────────────────
# 1. INFOS VIDÉO
# ─────────────────────────────────────────────────────────────
def get_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except:
        duration = 300.0
    try:
        size_mb = int(Path(video_path).stat().st_size / 1024 / 1024)
    except:
        size_mb = 0
    return {"duration": duration, "size_mb": size_mb}

# ─────────────────────────────────────────────────────────────
# 2. EXTRACTION AUDIO COMPLÈTE (fenêtres 5s — rapide)
# ─────────────────────────────────────────────────────────────
def extract_audio_energy_full(video_path: str, segment_duration: float = 5.0):
    """
    Analyse audio complète de la vidéo entière par fenêtres de 5s.
    Même pour 2h, ça prend ~2 minutes.
    """
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except:
        duration = 300.0

    cmd_audio = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "-"
    ]
    result_audio = subprocess.run(cmd_audio, capture_output=True, timeout=600)

    if len(result_audio.stdout) < 100:
        n = max(10, int(duration / segment_duration))
        return np.linspace(0, duration, n), np.ones(n) * 0.5, np.ones(n) * 0.5, duration

    samples = np.frombuffer(result_audio.stdout, dtype=np.float32)
    sr  = 16000
    hop = int(segment_duration * sr)

    times, energies, emotions = [], [], []
    for i in range(0, len(samples) - hop, hop):
        chunk = samples[i:i+hop]
        rms   = float(np.sqrt(np.mean(chunk**2)))
        zcr   = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
        var   = float(np.var(chunk))
        emotion = float(np.clip(zcr * 3 + var * 10, 0, 1))
        times.append(i / sr)
        energies.append(rms)
        emotions.append(emotion)

    return np.array(times), np.array(energies), np.array(emotions), duration

# ─────────────────────────────────────────────────────────────
# 3. DÉTECTION SCÈNES SUR VIDÉO COMPRESSÉE
# ─────────────────────────────────────────────────────────────
def compress_for_scene_detection(video_path: str) -> str:
    """Compresse à 360p/2fps juste pour détecter les scènes rapidement."""
    out_path = str(UPLOAD_DIR / f"scenes_{Path(video_path).stem}.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "scale=-2:360,fps=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
        "-an", out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0 and Path(out_path).exists():
        return out_path
    return str(video_path)

def extract_scene_changes(video_path: str) -> dict:
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", "select='gt(scene,0.25)',metadata=print:file=-",
        "-an", "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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
# 4. PRÉ-FILTRAGE : trouve les zones chaudes SANS Whisper
# ─────────────────────────────────────────────────────────────
def find_hot_zones(times, energies, emotions, scene_scores, duration, zone_duration=210):
    """
    Trouve les zones candidates (passages hype) via audio + scènes.
    Retourne une liste de (start, end) des zones les plus actives.
    zone_duration = taille de chaque zone à extraire pour Whisper (3.5 min)
    """
    # Normalise
    norm_audio = energies / energies.max() if energies.max() > 0 else np.ones_like(energies)
    norm_emo   = emotions / emotions.max() if emotions.max() > 0 else np.zeros_like(emotions)

    scene_boost = np.zeros(len(times))
    for st, sc in scene_scores.items():
        idx = np.argmin(np.abs(times - st))
        if idx < len(scene_boost):
            scene_boost[idx] += sc
    if scene_boost.max() > 0:
        scene_boost /= scene_boost.max()

    combined = norm_audio * 0.65 + scene_boost * 0.25 + norm_emo * 0.10

    # Lissage
    window   = max(5, min(20, int(len(times) / 40)))
    smoothed = np.convolve(combined, np.ones(window) / window, mode='same')

    # Découpe la vidéo en blocs de zone_duration secondes
    # et calcule le score moyen de chaque bloc
    step     = zone_duration
    n_blocks = max(1, int(duration / step))
    blocks   = []

    for b in range(n_blocks):
        b_start = b * step
        b_end   = min(b_start + zone_duration, duration)
        mask    = (times >= b_start) & (times < b_end)
        if mask.any():
            score = float(smoothed[mask].mean())
            blocks.append((b_start, b_end, score))

    # Trie par score décroissant — prend TOUS les blocs au-dessus du seuil
    blocks.sort(key=lambda x: x[2], reverse=True)
    threshold = np.percentile([b[2] for b in blocks], 40)  # top 60% des blocs
    hot = [(s, e) for s, e, sc in blocks if sc >= threshold]

    # Trie par ordre chronologique
    hot.sort(key=lambda x: x[0])
    return hot, smoothed

# ─────────────────────────────────────────────────────────────
# 5. EXTRACTION DE SEGMENTS AUDIO POUR WHISPER
# ─────────────────────────────────────────────────────────────
def extract_audio_segment(video_path: str, start: float, end: float, out_path: str) -> bool:
    """Extrait un segment audio en WAV mono 16kHz pour Whisper."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0 and Path(out_path).exists()

# ─────────────────────────────────────────────────────────────
# 6. WHISPER SUR LES ZONES CHAUDES UNIQUEMENT
# ─────────────────────────────────────────────────────────────
def transcribe_hot_zones(video_path: str, hot_zones: list, job_id: str) -> list:
    """
    Transcrit UNIQUEMENT les zones chaudes avec Whisper tiny.
    Ex: 10 zones × 3.5 min = 35 min audio → Whisper tiny = ~8 min de traitement
    """
    try:
        import whisper
        model = whisper.load_model("tiny")
    except Exception as e:
        print(f"Whisper non disponible: {e}")
        return []

    all_segments = []
    total = len(hot_zones)

    for i, (zone_start, zone_end) in enumerate(hot_zones):
        pct = 35 + int((i / total) * 20)
        update_job(job_id, progress=pct,
                   message=f"🧠 Whisper zone {i+1}/{total} ({int(zone_start//60)}min → {int(zone_end//60)}min)...")

        # Extrait l'audio de cette zone
        tmp_wav = str(UPLOAD_DIR / f"zone_{job_id}_{i}.wav")
        ok = extract_audio_segment(video_path, zone_start, zone_end, tmp_wav)
        if not ok:
            continue

        try:
            result = model.transcribe(
                tmp_wav,
                language=None,
                fp16=False,
                verbose=False,
            )
            for seg in result.get("segments", []):
                # Remet les timestamps dans le référentiel de la vidéo entière
                all_segments.append({
                    "start":       float(seg["start"]) + zone_start,
                    "end":         float(seg["end"])   + zone_start,
                    "text":        seg.get("text", "").strip(),
                    "viral_score": compute_semantic_score(seg.get("text", "").lower()),
                })
        except Exception as e:
            print(f"Whisper erreur zone {i}: {e}")
        finally:
            try:
                os.remove(tmp_wav)
            except:
                pass

    return all_segments

def compute_semantic_score(text: str) -> float:
    score = 0.0
    for word, weight in VIRAL_WORDS.items():
        if word in text:
            score += weight
    score += text.count("!") * 0.3
    score += text.count("?") * 0.15
    return min(score, 1.0)

def whisper_to_timeline(segments: list, times: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(times))
    for seg in segments:
        mask = (times >= seg["start"]) & (times <= seg["end"])
        if mask.any():
            scores[mask] = max(scores[mask].max(), seg["viral_score"])
    return scores

# ─────────────────────────────────────────────────────────────
# 7. SCORE VIRAL FINAL (avec Whisper si dispo)
# ─────────────────────────────────────────────────────────────
def compute_final_viral_score(times, energies, emotions, scene_scores, whisper_segments, duration):
    norm_audio = energies / energies.max() if energies.max() > 0 else np.ones_like(energies) * 0.5
    norm_emo   = emotions / emotions.max() if emotions.max() > 0 else np.zeros_like(emotions)

    scene_boost = np.zeros(len(times))
    for st, sc in scene_scores.items():
        idx = np.argmin(np.abs(times - st))
        if idx < len(scene_boost):
            scene_boost[idx] += sc
    if scene_boost.max() > 0:
        scene_boost /= scene_boost.max()

    whisper_scores = whisper_to_timeline(whisper_segments, times)

    if whisper_scores.max() > 0:
        # Avec Whisper : 40% sémantique + 35% audio + 15% scènes + 10% émotions
        combined = (
            whisper_scores * 0.40 +
            norm_audio     * 0.35 +
            scene_boost    * 0.15 +
            norm_emo       * 0.10
        )
    else:
        # Sans Whisper : 70% audio + 20% scènes + 10% émotions
        combined = (
            norm_audio  * 0.70 +
            scene_boost * 0.20 +
            norm_emo    * 0.10
        )

    window   = max(5, min(15, int(len(times) / 50)))
    smoothed = np.convolve(combined, np.ones(window) / window, mode='same')
    return smoothed

# ─────────────────────────────────────────────────────────────
# 8. DÉTECTION CLIPS — TOUS les passages hype, 61s à 180s
# ─────────────────────────────────────────────────────────────
def find_viral_clips(times, smoothed, duration):
    """
    Détecte TOUS les passages hype sans limite de nombre.
    Chaque clip : min 61s, max 180s, gap minimum 120s entre clips.
    """
    peak_threshold   = np.percentile(smoothed, 80)  # top 20% = passages hype
    extend_threshold = np.percentile(smoothed, 55)

    # Trouve tous les pics locaux
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= peak_threshold:
            if smoothed[i] >= smoothed[i-1] and smoothed[i] >= smoothed[i+1]:
                peaks.append(i)

    # Trie par score décroissant
    peaks.sort(key=lambda i: smoothed[i], reverse=True)
    clips = []

    for peak_idx in peaks:
        peak_time = times[peak_idx]

        # Vérifie le gap minimum avec les clips déjà trouvés
        if any(abs(peak_time - s) < MIN_GAP for s, e in clips):
            continue

        # Étend vers la gauche
        left = peak_idx
        while left > 0:
            if times[peak_idx] - times[left - 1] > MAX_DURATION / 2:
                break
            if smoothed[left - 1] >= extend_threshold:
                left -= 1
            else:
                break

        # Étend vers la droite
        right = peak_idx
        while right < len(times) - 1:
            if times[right + 1] - times[left] > MAX_DURATION:
                break
            if smoothed[right + 1] >= extend_threshold:
                right += 1
            else:
                if times[right] - times[left] < MIN_DURATION:
                    right += 1
                else:
                    break

        start_t = float(times[left])
        end_t   = float(times[right])

        # Garantit la durée minimale de 61s
        if end_t - start_t < MIN_DURATION:
            end_t = min(start_t + MIN_DURATION, duration)

        # Garantit de ne pas dépasser la vidéo
        if end_t > duration:
            end_t   = duration
            start_t = max(0, end_t - MIN_DURATION)

        # Garantit la durée maximale de 180s
        if end_t - start_t > MAX_DURATION:
            end_t = start_t + MAX_DURATION

        mask = (times >= start_t) & (times <= end_t)
        viral_score = float(smoothed[mask].mean()) if mask.any() else 0.5

        clips.append({
            "start":       round(start_t, 1),
            "end":         round(end_t, 1),
            "duration":    round(end_t - start_t, 1),
            "viral_score": round(viral_score, 3),
        })

    # Trie chronologiquement
    clips.sort(key=lambda x: x["start"])
    return clips

# ─────────────────────────────────────────────────────────────
# 9. CAPTION TIKTOK
# ─────────────────────────────────────────────────────────────
def generate_tiktok_caption(clip: dict, whisper_segments: list) -> str:
    clip_texts = [
        s["text"] for s in whisper_segments
        if s["start"] >= clip["start"] and s["end"] <= clip["end"]
    ]
    found_viral = []
    for text in clip_texts:
        for word in VIRAL_WORDS:
            if word in text.lower() and word not in found_viral:
                found_viral.append(word)

    score_pct = int(clip["viral_score"] * 100)

    if score_pct >= 80:
        templates = [
            "POV : tu tombes sur le moment le plus fou 🔥",
            "Ce moment va te laisser sans voix 😱",
            "Ils ont pas coupé ça au montage... 👀",
            "Le moment que tout le monde attendait 💥",
            "Je peux pas croire qu'ils ont dit ça 😳",
            "Ce passage va faire le buzz 🔥",
        ]
    elif score_pct >= 60:
        templates = [
            "Ce passage mérite vraiment d'être vu 👇",
            "Le meilleur moment de la vidéo 🎯",
            "Regarde jusqu'à la fin, ça vaut le coup 🔥",
            "Ce moment-là, on en parle ? 👀",
            "T'as vu ce moment ? 😤",
        ]
    else:
        templates = [
            "Moment clé à ne pas rater 👇",
            "À voir absolument 🎬",
            "Ce passage change tout 💡",
            "Extrait du meilleur moment 🎯",
        ]

    caption  = random.choice(templates)
    hashtags = "#viral #tiktok #fyp #pourtoi #clipviral"
    if found_viral:
        hashtags += " #" + found_viral[0].replace(" ", "")

    return f"{caption}\n\n{hashtags}"

# ─────────────────────────────────────────────────────────────
# 10. EXPORT CLIP TIKTOK (1080x1920, 61-180s)
# ─────────────────────────────────────────────────────────────
def export_clip_tiktok(input_path: str, start: float, end: float, out_path: str) -> bool:
    duration   = end - start
    fade_start = max(0, duration - 2.0)

    vf = (
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,boxblur=20:20,eq=brightness=-0.4[bg];"
        f"[0:v]scale=1080:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0

# ─────────────────────────────────────────────────────────────
# 11. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────
def process_video_job(job_id: str, video_path: str):
    job_dir        = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    scene_vid_path = None

    try:
        # ÉTAPE 1 — Infos
        update_job(job_id, status="analyzing", progress=3,
                   message="📋 Lecture de la vidéo...")
        info     = get_video_info(video_path)
        duration = info["duration"]
        size_mb  = info["size_mb"]
        mins     = int(duration // 60)
        secs     = int(duration % 60)
        update_job(job_id, progress=5,
                   message=f"📹 Vidéo : {mins}min {secs}s · {size_mb} MB détectés")

        # ÉTAPE 2 — Analyse audio complète (rapide, fenêtres 5s)
        update_job(job_id, progress=8,
                   message="🎵 Analyse audio complète de la vidéo...")
        times, energies, emotions, _ = extract_audio_energy_full(video_path, segment_duration=5.0)

        # ÉTAPE 3 — Compression pour détection de scènes
        update_job(job_id, progress=22,
                   message="🎬 Compression pour détection de scènes...")
        scene_vid_path = compress_for_scene_detection(video_path)

        update_job(job_id, progress=28,
                   message="🎬 Détection des changements de scène...")
        scene_scores = extract_scene_changes(scene_vid_path)

        # ÉTAPE 4 — Pré-filtrage : trouve les zones chaudes
        update_job(job_id, progress=33,
                   message="🔥 Identification des zones les plus hype...")
        hot_zones, pre_smoothed = find_hot_zones(
            times, energies, emotions, scene_scores, duration, zone_duration=210
        )
        nb_zones = len(hot_zones)
        total_whisper_min = round(sum(e - s for s, e in hot_zones) / 60, 1)
        update_job(job_id, progress=35,
                   message=f"✅ {nb_zones} zones hype trouvées ({total_whisper_min} min à analyser)")

        # ÉTAPE 5 — Whisper UNIQUEMENT sur les zones chaudes
        whisper_segments = transcribe_hot_zones(video_path, hot_zones, job_id)
        nb_seg = len(whisper_segments)
        update_job(job_id, progress=55,
                   message=f"✍️ {nb_seg} segments transcrits. Calcul du score viral...")

        # ÉTAPE 6 — Score viral final combiné
        smoothed = compute_final_viral_score(
            times, energies, emotions, scene_scores, whisper_segments, duration
        )
        clips = find_viral_clips(times, smoothed, duration)

        if not clips:
            update_job(job_id, status="error",
                       message="❌ Aucun passage hype détecté. Essaie avec une autre vidéo.")
            return

        update_job(job_id, progress=60, status="exporting",
                   message=f"✅ {len(clips)} passages hype détectés ! Export TikTok 1080×1920...")

        # ÉTAPE 7 — Export depuis vidéo ORIGINALE (qualité maximale)
        exported = []
        for idx, clip in enumerate(clips):
            out_path = job_dir / f"Clip_Elite_{idx+1}.mp4"
            caption  = generate_tiktok_caption(clip, whisper_segments)
            success  = export_clip_tiktok(video_path, clip["start"], clip["end"], str(out_path))

            if success:
                exported.append({
                    **clip,
                    "filename":    f"Clip_Elite_{idx+1}.mp4",
                    "url":         f"/outputs/{job_id}/Clip_Elite_{idx+1}.mp4",
                    "preview_url": f"/outputs/{job_id}/Clip_Elite_{idx+1}.mp4",
                    "rank":        idx + 1,
                    "caption":     caption,
                })

            progress = 60 + int(((idx + 1) / len(clips)) * 38)
            update_job(job_id, progress=progress,
                       message=f"🎬 Export clip {idx+1}/{len(clips)} ({int(clip['duration'])}s)...")

        update_job(job_id, status="done", progress=100, clips=exported,
                   message=f"🎉 {len(exported)} clips viraux prêts à poster !")

    except Exception as e:
        update_job(job_id, status="error", message=f"❌ Erreur : {str(e)}")
    finally:
        try:
            os.remove(video_path)
        except:
            pass
        if scene_vid_path:
            try:
                os.remove(scene_vid_path)
            except:
                pass

# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_video(request: Request, background_tasks: BackgroundTasks):
    job_id   = str(uuid.uuid4())[:8]
    content_disposition = request.headers.get("content-disposition", "")
    filename = "video.mp4"
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[-1].strip('"').strip("'")

    ext        = Path(filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{ext}"

    try:
        with open(video_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
    except Exception as e:
        raise HTTPException(500, f"Erreur upload : {e}")

    if not video_path.exists() or video_path.stat().st_size < 1000:
        raise HTTPException(400, "Fichier vide ou invalide.")

    jobs[job_id] = {
        "status":     "queued",
        "progress":   0,
        "message":    "✅ Vidéo reçue, analyse en cours...",
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


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in ws_connections:
        ws_connections[job_id] = []
    ws_connections[job_id].append(websocket)
    if job_id in jobs:
        await websocket.send_json(jobs[job_id])
    try:
        while True:
            await asyncio.sleep(15)
            try:
                await websocket.send_json({"ping": True})
            except:
                break
            if job_id in jobs and jobs[job_id].get("status") in ("done", "error"):
                await asyncio.sleep(2)
                break
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in ws_connections and websocket in ws_connections[job_id]:
            ws_connections[job_id].remove(websocket)


@app.get("/outputs/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, "Clip introuvable.")
    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        filename=filename,
        headers={
            "Accept-Ranges":       "bytes",
            "Content-Disposition": f"inline; filename={filename}",
        }
    )


# Serve frontend
static_path = Path("static")
if static_path.exists() and not static_path.is_dir():
    static_path.unlink()
static_path.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
