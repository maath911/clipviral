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

# ─── Job storage (in-memory, replace with Redis for prod) ───
jobs = {}

# ─── CONFIG (same as robot.py) ───
MIN_DURATION = 61
MAX_DURATION = 180

# ─────────────────────────────────────────────────────────────
# VIRAL DETECTION ENGINE (adapted from robot.py)
# Uses audio energy analysis as proxy for YouTube heatmap
# ─────────────────────────────────────────────────────────────

def extract_audio_energy(video_path: str, segment_duration: float = 2.0) -> tuple:
    """
    Extract audio RMS energy per time segment using FFmpeg.
    This replicates the YouTube heatmap: high energy = most watched moment.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    duration = float(info["format"]["duration"])
    
    # Extract audio as raw PCM samples
    cmd_audio = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "22050",
        "-f", "f32le", "-"
    ]
    result_audio = subprocess.run(cmd_audio, capture_output=True)
    
    if len(result_audio.stdout) < 100:
        # No audio, use scene detection only
        return np.linspace(0, duration, 100), np.ones(100) * 0.5, duration

    samples = np.frombuffer(result_audio.stdout, dtype=np.float32)
    sr = 22050
    hop = int(segment_duration * sr)
    
    times = []
    energies = []
    
    for i in range(0, len(samples) - hop, hop):
        chunk = samples[i:i+hop]
        rms = np.sqrt(np.mean(chunk**2))
        times.append(i / sr)
        energies.append(float(rms))
    
    return np.array(times), np.array(energies), duration


def extract_scene_changes(video_path: str) -> dict:
    """
    Detect scene changes using FFmpeg scene filter.
    Returns a dict of {timestamp: score}.
    """
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


def compute_viral_score(times, audio_energies, scene_scores, duration):
    """
    Combine audio energy + scene changes into a viral score per segment.
    Same peak-detection algorithm as robot.py heatmap analysis.
    """
    # Normalize audio energies
    if audio_energies.max() > 0:
        norm_audio = audio_energies / audio_energies.max()
    else:
        norm_audio = np.ones_like(audio_energies) * 0.5
    
    # Add scene change boost to nearest time segments
    scene_boost = np.zeros(len(times))
    for scene_time, scene_score in scene_scores.items():
        idx = np.argmin(np.abs(times - scene_time))
        if idx < len(scene_boost):
            scene_boost[idx] += scene_score * 0.3  # weight scene changes at 30%
    
    # Combined score: 70% audio energy + 30% scene changes
    combined = norm_audio * 0.7 + np.clip(scene_boost, 0, 1) * 0.3
    
    # Smooth (same 5-point moving average as robot.py)
    window = 7
    smoothed = np.convolve(combined, np.ones(window)/window, mode='same')
    
    return smoothed


def find_viral_clips(times, smoothed, duration):
    """
    Exact same peak detection algorithm from robot.py get_most_viewed_only().
    """
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

        # Extend left
        left = peak_idx
        while left > 0:
            if times[peak_idx] - times[left - 1] > MAX_DURATION / 2:
                break
            if smoothed[left - 1] >= extend_threshold:
                left -= 1
            else:
                break

        # Extend right
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
        clip_duration = end_t - start_t

        if clip_duration < MIN_DURATION:
            end_t = min(start_t + MIN_DURATION, duration)
            clip_duration = end_t - start_t

        if end_t > duration:
            end_t = duration
            start_t = max(0, end_t - MIN_DURATION)

        # Viral score = mean smoothed value over the clip
        mask = (times >= start_t) & (times <= end_t)
        viral_score = float(smoothed[mask].mean()) if mask.any() else 0.5
        
        clips.append({
            "start": round(start_t, 1),
            "end": round(end_t, 1),
            "duration": round(end_t - start_t, 1),
            "viral_score": round(viral_score, 3),
        })

    clips.sort(key=lambda x: x["start"])
    return clips


def export_clip_tiktok(input_path: str, start: float, end: float, out_path: str):
    """
    Export a single clip in TikTok format (1080x1920, vertical).
    Exact same FFmpeg pipeline as robot.py export_clip_ffmpeg().
    """
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────
# BACKGROUND PROCESSING
# ─────────────────────────────────────────────────────────────

def process_video_job(job_id: str, video_path: str):
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    try:
        jobs[job_id]["status"] = "analyzing"
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"] = "Analyse de l'énergie audio..."

        times, energies, duration = extract_audio_energy(video_path)
        jobs[job_id]["progress"] = 30
        jobs[job_id]["message"] = "Détection des changements de scène..."

        scene_scores = extract_scene_changes(video_path)
        jobs[job_id]["progress"] = 50
        jobs[job_id]["message"] = "Calcul du score viral..."

        smoothed = compute_viral_score(times, energies, scene_scores, duration)
        clips    = find_viral_clips(times, smoothed, duration)

        if not clips:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = "Aucun clip viral détecté."
            return

        jobs[job_id]["clips_meta"] = clips
        jobs[job_id]["progress"]   = 55
        jobs[job_id]["message"]    = f"✅ {len(clips)} clips trouvés. Export TikTok..."
        jobs[job_id]["status"]     = "exporting"

        exported = []
        for idx, clip in enumerate(clips):
            out_path = job_dir / f"Clip_Elite_{idx+1}.mp4"
            success  = export_clip_tiktok(video_path, clip["start"], clip["end"], str(out_path))
            
            if success:
                exported.append({
                    **clip,
                    "filename": f"Clip_Elite_{idx+1}.mp4",
                    "url": f"/outputs/{job_id}/Clip_Elite_{idx+1}.mp4",
                    "rank": idx + 1,
                })
            
            progress = 55 + int(((idx + 1) / len(clips)) * 40)
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
        # Cleanup uploaded video
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
        "status": "queued",
        "progress": 0,
        "message": "En attente...",
        "clips": [],
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
import os
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
