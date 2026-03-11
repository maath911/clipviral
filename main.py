# ═══════════════════════════════════════════════════════════════════
#  ClipViral FREE — Version 100% gratuite, sans IA, sans crash
#  ✅ Aucune API payante   ✅ Pas de Whisper   ✅ Tourne sur Render Free
#  Détection audio pure : énergie RMS + variance + silence detection
# ═══════════════════════════════════════════════════════════════════
import os
import gc
import glob
import uuid
import shutil
import subprocess
import numpy as np
import json
import time
import asyncio
import random
import zipfile
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ClipViral FREE API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config FREE — conservateur pour Render Free 512MB ───────────────
MAX_VIDEO_DURATION = 3600   # 60 min (même limite que la version PRO)
MAX_FILE_SIZE      = 2 * 1024 * 1024 * 1024  # 2 GB
CLIP_MIN  = 61
CLIP_MAX  = 180  # 61-180s comme demandé
CLIP_GAP  = 90

jobs: dict           = {}
ws_connections: dict = defaultdict(list)
ip_jobs: dict        = defaultdict(list)
MAX_JOBS_PER_IP      = 5
executor             = ThreadPoolExecutor(max_workers=1)

# ── Helpers ──────────────────────────────────────────────────────────
def set_job(job_id: str, **kw):
    if job_id in jobs:
        jobs[job_id].update(kw)

async def notify_ws(job_id: str):
    try:
        data = json.dumps(jobs.get(job_id, {}), default=str)
    except Exception:
        try:
            j = jobs.get(job_id, {})
            data = json.dumps({"status": j.get("status","unknown"),
                               "progress": j.get("progress",0), "message":"..."})
        except Exception:
            return
    dead = []
    for ws in list(ws_connections.get(job_id, [])):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for d in dead:
        try: ws_connections[job_id].remove(d)
        except ValueError: pass

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name).strip('_.')
    return name[:80] or "video"

# ── 1. Durée ─────────────────────────────────────────────────────────
def _get_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe","-v","quiet","-print_format","json","-show_format","-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise ValueError("Lecture trop lente.")
    if not r.stdout:
        raise ValueError("Fichier invalide.")
    info = json.loads(r.stdout)
    dur = None
    raw = info.get("format",{}).get("duration","N/A")
    if raw not in ("N/A","",None):
        try: dur = float(raw) if float(raw) > 0 else None
        except: pass
    if not dur:
        for s in info.get("streams",[]):
            raw = s.get("duration","N/A")
            if raw not in ("N/A","",None):
                try:
                    d = float(raw)
                    if d > 0: dur = d; break
                except: pass
    if not dur:
        raise ValueError("Impossible de lire la durée.")
    return float(dur)

# ── 2. Audio ─────────────────────────────────────────────────────────
def _has_audio(video_path: str) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe","-v","quiet","-print_format","json","-show_streams",str(video_path)],
            capture_output=True, text=True, timeout=15)
        info = json.loads(r.stdout)
        return any(s.get("codec_type")=="audio" for s in info.get("streams",[]))
    except Exception:
        return True

# ── 3. H264 ──────────────────────────────────────────────────────────
def _ensure_h264(video_path: str, job_id: str, known_duration: float = None) -> str:
    try:
        r = subprocess.run(
            ["ffprobe","-v","quiet","-print_format","json","-show_streams",str(video_path)],
            capture_output=True, text=True, timeout=30)
        streams = json.loads(r.stdout).get("streams",[])
        codec = next((s["codec_name"] for s in streams if s.get("codec_type")=="video"), "h264")
    except subprocess.TimeoutExpired:
        return video_path
    except Exception:
        codec = "h264"

    if codec.lower() not in {"hevc","h265","vp9","av1","vp8","wmv3","mpeg4","theora"}:
        return video_path

    dur = known_duration or 0
    if dur > 1200:  # > 20min en codec exotique → trop lent à ré-encoder
        raise ValueError(
            f"Ta vidéo utilise le codec {codec.upper()} et dure {int(dur//60)}min. "
            f"Convertis-la en MP4 H264 avec HandBrake (gratuit) avant de l'uploader.")

    out = str(Path(video_path).parent / f"{job_id}_h264.mp4")
    try:
        r2 = subprocess.run([
            "ffmpeg","-y","-i",str(video_path),
            "-c:v","libx264","-preset","ultrafast","-crf","26",
            "-c:a","aac","-b:a","128k","-movflags","+faststart", out
        ], capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        return video_path
    if r2.returncode == 0:
        try: os.remove(video_path)
        except: pass
        return out
    return video_path

# ── 4. Énergie audio — détection PURE sans IA ────────────────────────
def _extract_audio_energy(video_path: str, seg_dur: float = 3.0, duration: float = None):
    """
    Version FREE: détection audio pure.
    Analyse: énergie RMS + variance spectrale + détection silences.
    Pas de Whisper, pas d'IA — rapide et stable.
    """
    if not duration:
        duration = _get_duration(video_path)

    CHUNK = 300
    all_times, all_energies, all_emotions = [], [], []

    def _neutral(start, count):
        for i in range(count):
            all_times.append(start + i * seg_dur)
            all_energies.append(0.3)
            all_emotions.append(0.3)

    for chunk_start in range(0, int(duration), CHUNK):
        chunk_end = min(chunk_start + CHUNK, duration)
        chunk_dur = chunk_end - chunk_start
        n = max(1, int(chunk_dur / seg_dur))

        try:
            r = subprocess.run([
                "ffmpeg","-y",
                "-ss", str(chunk_start), "-i", str(video_path),
                "-t", str(chunk_dur),
                "-vn","-ac","1","-ar","22050","-f","f32le","-"
            ], capture_output=True, timeout=90)
        except subprocess.TimeoutExpired:
            _neutral(chunk_start, n)
            continue

        if not r.stdout or len(r.stdout) < 400:
            _neutral(chunk_start, n)
            continue

        try:
            samples = np.frombuffer(r.stdout, dtype=np.float32)
        except Exception:
            _neutral(chunk_start, n)
            continue

        if len(samples) < 100:
            _neutral(chunk_start, n)
            continue

        sr  = 22050
        hop = int(seg_dur * sr)

        for i in range(0, len(samples) - hop, hop):
            seg = samples[i:i+hop]
            rms      = float(np.sqrt(np.mean(seg**2)))
            variance = float(np.var(seg))
            # Détection de dynamique: changement d'énergie entre 1ère et 2ème moitié
            half = len(seg)//2
            delta = abs(float(np.sqrt(np.mean(seg[half:]**2))) - float(np.sqrt(np.mean(seg[:half]**2))))
            # Emotion = variance normalisée + dynamique
            emotion = min(1.0, variance * 40 + delta * 3)
            all_times.append(chunk_start + i/sr)
            all_energies.append(rms)
            all_emotions.append(emotion)

        del samples

    if not all_times:
        all_times, all_energies, all_emotions = [0.0], [0.3], [0.3]

    return np.array(all_times), np.array(all_energies), np.array(all_emotions), float(duration)

# ── 5. Détection clips ───────────────────────────────────────────────
def _auto_detect_clips(times, energies, emotions, duration) -> list:
    norm_e  = energies / (energies.max() + 1e-9)
    norm_em = emotions / (emotions.max() + 1e-9)
    combined = norm_e * 0.65 + norm_em * 0.35
    window   = max(3, int(10/3))
    smoothed = np.convolve(combined, np.ones(window)/window, mode="same")

    mean_s = smoothed.mean()
    std_s  = smoothed.std()
    cv     = std_s / (mean_s + 1e-6)
    k      = 0.8 if cv > 0.5 else (1.0 if cv > 0.3 else 0.5)
    threshold = mean_s + k * std_s

    if   duration <= 600:  max_clips = 2
    elif duration <= 900:  max_clips = 3
    elif duration <= 1800: max_clips = 5
    elif duration <= 3600: max_clips = 8
    else:                  max_clips = 10

    peaks = [i for i in range(1, len(smoothed)-1)
             if smoothed[i] >= threshold
             and smoothed[i] >= smoothed[i-1]
             and smoothed[i] >= smoothed[i+1]]
    peaks.sort(key=lambda i: smoothed[i], reverse=True)

    ext_thresh = mean_s + 0.2 * std_s
    clips = []

    for pi in peaks:
        if len(clips) >= max_clips: break
        peak_t = float(times[pi])
        if any(abs(peak_t - (c["start"]+c["end"])/2) < CLIP_GAP for c in clips):
            continue

        left = pi
        while left > 0 and (peak_t - times[left-1]) <= CLIP_MAX and smoothed[left-1] >= ext_thresh:
            left -= 1
        right = pi
        while right < len(times)-1:
            if (times[right+1] - times[left]) > CLIP_MAX: break
            if smoothed[right+1] >= ext_thresh: right += 1
            elif (times[right] - times[left]) < CLIP_MIN: right += 1
            else: break

        s = float(times[left])
        e = float(times[right])
        if (e-s) < CLIP_MIN: e = min(s+CLIP_MIN, duration)
        if (e-s) > CLIP_MAX:
            s = max(0.0, peak_t - CLIP_MAX/2)
            e = min(duration, s+CLIP_MAX)
        if e > duration:
            e = duration; s = max(0.0, e-CLIP_MIN)
        if e-s <= 0: continue

        mask = (times >= s) & (times <= e)
        vs = min(1.0, float(smoothed[mask].mean())/(smoothed.max()+1e-6)) if mask.any() else 0.5
        clips.append({"start":round(s,1),"end":round(e,1),"duration":round(e-s,1),
                      "viral_score":round(vs,3),"reason":""})

    if not clips:
        n = min(3, max(1, int(duration/600)))
        step = duration/(n+1)
        for i in range(n):
            s = max(0.0, round(step*(i+1)-CLIP_MIN/2, 1))
            e = min(float(duration), s+CLIP_MIN)
            if e-s > 0:
                clips.append({"start":round(s,1),"end":round(e,1),
                              "duration":round(e-s,1),"viral_score":0.5,"reason":"Fallback"})

    clips.sort(key=lambda x: x["start"])
    return clips

# ── 6. Score TikTok (heuristique pure) ──────────────────────────────
def _tiktok_score(clip: dict, duration: float) -> dict:
    dur  = clip["end"] - clip["start"]
    base = clip.get("viral_score", 0.5)
    total = 0.0
    det = {}

    if 45 <= dur <= 90:   ds,dt = 1.0,"✅ Durée parfaite TikTok (45-90s)"
    elif 30 <= dur < 45:  ds,dt = 0.8,"🟡 Un peu court — idéal 45-90s"
    elif 90 < dur <= 120: ds,dt = 0.8,"🟡 Un peu long — idéal 45-90s"
    elif dur < 30:        ds,dt = 0.5,"⚠️ Trop court pour TikTok"
    else:                 ds,dt = 0.6,"⚠️ Trop long — découpe en 2"
    det["duree"] = {"score":ds,"tip":dt}
    total += ds*30

    pos_ratio = clip["start"]/max(duration,1)
    if pos_ratio < 0.25:   ps,pt = 1.0,"✅ Moment en début de vidéo — fort potentiel"
    elif pos_ratio < 0.6:  ps,pt = 0.8,"🟡 Moment au milieu de la vidéo"
    else:                  ps,pt = 0.6,"⚠️ Moment en fin de vidéo — accroche plus difficile"
    det["position"] = {"score":ps,"tip":pt}
    total += ps*20

    det["energie"] = {"score":base,"tip":"✅ Énergie forte" if base>=0.7 else "🟡 Énergie correcte" if base>=0.4 else "⚠️ Énergie faible"}
    total += base*30

    det["conseil"] = {"score":0.7,"tip":"💡 Ajoute un texte d'accroche dans les 3 premières secondes"}
    total += 0.7*20

    pct = round(total)
    if   pct >= 85: grade,gc = "A","#00dfa2"
    elif pct >= 70: grade,gc = "B","#5b8dee"
    elif pct >= 55: grade,gc = "C","#f5c842"
    elif pct >= 40: grade,gc = "D","#ff8c00"
    else:           grade,gc = "F","#ff2d55"
    return {"grade":grade,"grade_color":gc,"tiktok_score":pct,"details":det}

# ── 7. Export TikTok 1080×1920 ──────────────────────────────────────
def _export_clip(input_path, start, end, out_path, watermark=None, has_audio=True) -> bool:
    dur  = end - start
    if dur <= 0: return False
    fade = max(0.0, dur-2.0)

    # ── Optimisé Render Free: 720×1280 + threads 1 + CRF 28 ──────────
    # 720p au lieu de 1080p → 2.25x moins de pixels → 2x plus rapide à encoder
    # TikTok re-encode de toute façon, 720p est indiscernable après compression
    # -threads 1 → stable sur vCPU partagé Render Free
    # CRF 28 → qualité parfaite pour TikTok (CRF 23 est du gaspillage CPU)
    W, H = 720, 1280

    simple_vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fade=t=out:st={fade}:d=2"
    )

    if watermark:
        safe_wm = (watermark.replace("\\","\\\\").replace("'","\\'")
                            .replace(":","\\:").replace("%","\\%"))
        full_vf = (
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fade=t=out:st={fade}:d=2[vbase];"
            f"[vbase]drawtext=text='{safe_wm}':fontsize=20:fontcolor=white@0.65:x=12:y=12[vout]"
        )
        cmd = ["ffmpeg","-y","-ss",str(start),"-i",str(input_path),"-t",str(dur),
               "-filter_complex",full_vf,"-map","[vout]","-threads","1"]
    else:
        cmd = ["ffmpeg","-y","-ss",str(start),"-i",str(input_path),"-t",str(dur),
               "-vf",simple_vf,"-map","0:v","-threads","1"]

    if has_audio:
        cmd += ["-map","0:a?","-af",f"afade=t=out:st={fade}:d=2","-c:a","aac","-b:a","128k"]
    cmd += ["-c:v","libx264","-preset","ultrafast","-crf","28",
            "-pix_fmt","yuv420p","-movflags","+faststart",str(out_path)]
    try:
        # 4min max par clip — si ça dépasse, le clip est skippé (pas de crash)
        r = subprocess.run(cmd, capture_output=True, timeout=240)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

# ── 8. Titre propre ─────────────────────────────────────────────────────────
def _make_title(clip: dict, rank: int, video_title: str = "") -> str:
    s = int(clip["start"] // 60); sm = int(clip["start"] % 60)
    e = int(clip["end"] // 60);   em = int(clip["end"] % 60)
    vs = int(clip.get("viral_score", 0.5) * 100)
    label = "Moment Viral" if vs >= 80 else "Moment Fort" if vs >= 65 else "Moment Clé" if vs >= 50 else "Extrait"
    timing = f"{s}:{sm:02d}→{e}:{em:02d}"
    if video_title:
        short = video_title[:30] + ("…" if len(video_title) > 30 else "")
        return f"{label} #{rank} | {short} ({timing})"
    return f"{label} #{rank} ({timing}) · {int(clip['duration'])}s"

def _make_hashtags(clip: dict, video_title: str = "") -> list:
    base = ["#fyp","#pourtoi","#viral","#tiktok","#foryou"]
    t = video_title.lower()
    if any(k in t for k in ["basket","nba","lakers","celtics","spurs","rockets","wemby","lebron","curry","nfl","foot","soccer","goal","sport","highlight"]):
        extra = ["#sport","#nba","#basketball","#highlights","#sports"]
    elif any(k in t for k in ["gaming","game","fortnite","minecraft","valorant","warzone","cod","fifa","streamer","twitch"]):
        extra = ["#gaming","#gamer","#gameplay","#streamer","#game"]
    elif any(k in t for k in ["music","musique","rap","song","clip","album","concert","live","freestyle"]):
        extra = ["#music","#musique","#rap","#newmusic","#song"]
    elif any(k in t for k in ["react","reaction","funny","humour","wtf","fail","best","top","drôle"]):
        extra = ["#reaction","#funny","#humour","#wtf","#bestmoment"]
    elif any(k in t for k in ["news","actu","politique","info","france","monde"]):
        extra = ["#actu","#news","#france","#info","#polemique"]
    elif any(k in t for k in ["food","recette","cuisine","recipe","chef","eat"]):
        extra = ["#food","#cuisine","#recette","#foodtiktok","#chef"]
    else:
        vs = clip.get("viral_score", 0.5)
        extra = ["#choc","#incroyable","#wow","#incredible","#mustwatch"] if vs >= 0.75 else ["#interesting","#watch","#explore","#content","#trending"]
    return base + extra[:5]

# ── 8. Caption ───────────────────────────────────────────────────────
def _make_caption(clip: dict) -> str:
    vs = int(clip.get("viral_score",0.5)*100)
    if vs >= 80:
        t = random.choice([
            "POV : tu tombes sur le moment le plus fou 🔥",
            "Ce moment va te laisser sans voix 😱",
            "Ils ont pas coupé ça au montage... 👀",
            "Le moment que tout le monde attendait 💥",
        ])
    elif vs >= 60:
        t = random.choice([
            "Ce passage mérite d'être vu 👇",
            "Le meilleur moment de la vidéo 🎯",
            "Regarde jusqu'à la fin 🔥",
        ])
    else:
        t = random.choice(["Moment clé à ne pas rater 👇","À voir absolument 🎬"])
    return f"{t}\n\n#viral #tiktok #fyp #pourtoi"

# ── 9. PIPELINE FREE ─────────────────────────────────────────────────
async def process_video_job(job_id, video_path, filename, settings=None):
    if settings is None: settings = {}
    job_dir   = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    loop      = asyncio.get_running_loop()
    watermark = settings.get("watermark","").strip()[:40]

    async def upd(pct, msg, **kw):
        set_job(job_id, progress=pct, message=msg, **kw)
        await notify_ws(job_id)

    try:
        size_mb = Path(video_path).stat().st_size // (1024*1024)
        title   = Path(filename).stem[:60]
        set_job(job_id, video_title=title)
        await upd(5, f"✅ Vidéo reçue ({size_mb} MB) — vérification...")

        duration = float(settings.get("_known_duration") or
                         await loop.run_in_executor(executor, _get_duration, video_path))

        if duration > MAX_VIDEO_DURATION:
            m = int(duration//60)
            s = int(duration%60)
            raise ValueError(
                f"Vidéo trop longue : {m}min {s}s détectées. Maximum 60min. Découpe avec VLC puis réenvoie.")

        mins = int(duration//60)
        await upd(10, f"⏱️ Durée : {mins}min. Vérification format...")

        video_path = await loop.run_in_executor(executor, _ensure_h264, video_path, job_id, duration)
        await upd(18, "🔧 Format OK. Analyse audio...")

        # Étape principale: analyse audio pure (pas de Whisper → jamais de freeze)
        await upd(22, "🎵 Analyse de l'énergie audio en cours...")

        # Heartbeat asyncio: envoie une update toutes les 25s pendant l'analyse audio
        # pour que le client sache que le job est vivant (l'analyse peut prendre 1-5min)
        audio_done = asyncio.Event()
        async def _audio_heartbeat():
            icons = ["🎵","🎶","🎤","🔊","🎵"]
            i = 0
            pct = 22
            while not audio_done.is_set():
                await asyncio.sleep(25)
                if audio_done.is_set(): break
                pct = min(54, pct + 4)  # monte doucement de 22 vers 54
                ico = icons[i % len(icons)]
                await upd(pct, f"{ico} Analyse audio en cours... ({pct}%)")
                i += 1
        hb_task = asyncio.create_task(_audio_heartbeat())

        try:
            times, energies, emotions, duration = await loop.run_in_executor(
                executor, _extract_audio_energy, video_path, 3.0, duration)
        finally:
            audio_done.set()
            hb_task.cancel()
            try: await hb_task
            except asyncio.CancelledError: pass

        await upd(55, f"✅ Audio analysé ({mins}min). Détection des moments forts...")

        clips = await loop.run_in_executor(
            executor, _auto_detect_clips, times, energies, emotions, duration)
        if not clips:
            raise ValueError("Aucun moment fort détecté. Essaie une vidéo avec plus de variations audio.")

        await upd(65, f"✅ {len(clips)} moment(s) détecté(s). Export TikTok...", status="exporting")

        has_audio = await loop.run_in_executor(executor, _has_audio, video_path)
        exported  = []

        for i, clip in enumerate(clips):
            name     = f"Clip_Elite_{i+1}"
            out_path = str(job_dir / f"{name}.mp4")
            dur_s    = int(clip["duration"])
            pct_base = 65 + int(i/len(clips)*32)

            # Heartbeat pendant l'export du clip (max 3min sur Render Free)
            export_done = asyncio.Event()
            async def _export_heartbeat(pb=pct_base, n=i+1, tot=len(clips), ds=dur_s):
                secs = 0
                while not export_done.is_set():
                    await asyncio.sleep(12)
                    if export_done.is_set(): break
                    secs += 12
                    await upd(pb, f"🎬 Export clip {n}/{tot} ({ds}s) — {secs}s écoulées...")
            hb_exp = asyncio.create_task(_export_heartbeat())

            try:
                ok = await loop.run_in_executor(
                    executor, _export_clip, video_path,
                    clip["start"], clip["end"], out_path, watermark or None, has_audio)
            finally:
                export_done.set()
                hb_exp.cancel()
                try: await hb_exp
                except asyncio.CancelledError: pass

            if ok:
                score = _tiktok_score(clip, duration)
                # Filtre: on garde seulement A et B (crème de la crème)
                # Si tous les clips sont C/D/F → on garde quand même le meilleur
                clip_grade = score["grade"]
                clip_rank_actual = i + 1
                hashtags = _make_hashtags(clip, title)
                clip_title = _make_title(clip, clip_rank_actual, title)
                caption_base = _make_caption(clip)
                caption_full = f"{caption_base}\n\n{' '.join(hashtags[:8])}"
                exported.append({
                    **clip,
                    "filename":     f"{name}.mp4",
                    "url":          f"/outputs/{job_id}/{name}.mp4",
                    "preview_url":  f"/outputs/{job_id}/{name}.mp4",
                    "rank":         clip_rank_actual,
                    "clip_title":   clip_title,
                    "caption":      caption_full,
                    "hashtags":     hashtags,
                    "video_title":  title,
                    "tiktok_score": score["tiktok_score"],
                    "grade":        clip_grade,
                    "grade_color":  score["grade_color"],
                    "tt_details":   score["details"],
                    "has_subtitles": False,
                })
            pct = 65 + int((i+1)/len(clips)*32)
            await upd(pct, f"✅ Clip {i+1}/{len(clips)} exporté ({dur_s}s).")

        if not exported:
            raise ValueError("Export échoué. Réessaie avec une autre vidéo.")

        # Filtre qualité : garde seulement A et B
        GRADES_OK = {"A", "B"}
        premium = [c for c in exported if c.get("grade") in GRADES_OK]
        if premium:
            # Re-numérote les clips gardés
            for idx2, c in enumerate(premium):
                c["rank"] = idx2 + 1
                c["clip_title"] = _make_title(c, idx2 + 1, title)
            exported = premium
        # Si aucun A/B → garde quand même le meilleur (évite résultat vide)
        # exported reste inchangé avec tous les clips

        zip_path = job_dir / "tous_les_clips.zip"
        def _make_zip():
            with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_STORED) as zf:
                for c in exported:
                    p = job_dir / c["filename"]
                    if p.exists(): zf.write(p, c["filename"])
        await loop.run_in_executor(executor, _make_zip)
        zip_url = f"/outputs/{job_id}/tous_les_clips.zip" if zip_path.exists() else None

        set_job(job_id, status="done", progress=100, clips=exported,
                zip_url=zip_url, message=f"🎉 {len(exported)} clips prêts !")
        await notify_ws(job_id)

    except Exception as e:
        set_job(job_id, status="error", message=f"❌ {str(e)}")
        await notify_ws(job_id)
    finally:
        for p in [video_path, str(UPLOAD_DIR/f"{job_id}.mp4"), str(UPLOAD_DIR/f"{job_id}_h264.mp4")]:
            try:
                if Path(p).exists(): os.remove(p)
            except: pass

# ── WebSocket ────────────────────────────────────────────────────────
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    ws_connections[job_id].append(websocket)
    try:
        if job_id in jobs:
            await websocket.send_text(json.dumps(jobs[job_id], default=str))
        while True:
            await asyncio.sleep(20)
            try: await websocket.send_text('{"ping":true}')
            except: break
    except WebSocketDisconnect: pass
    except: pass
    finally:
        try: ws_connections[job_id].remove(websocket)
        except ValueError: pass

# ── Upload ───────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    ip_jobs[client_ip] = [t for t in ip_jobs[client_ip] if now-t < 3600]
    if len(ip_jobs[client_ip]) >= MAX_JOBS_PER_IP:
        raise HTTPException(429, "Trop de vidéos envoyées. Attends 1h.")
    ip_jobs[client_ip].append(now)

    job_id   = str(uuid.uuid4())[:8]
    tmp_path = str(UPLOAD_DIR / f"{job_id}.mp4")

    cd = request.headers.get("content-disposition","")
    filename = "video.mp4"
    if "filename=" in cd:
        try:
            raw = cd.split("filename=")[1].strip().strip('"\'')
            filename = sanitize_filename(raw) + ".mp4"
        except: pass

    written = 0
    first_chunk = True
    # Boxes MP4 valides en position 4-8 — liste élargie pour couvrir tous les encodeurs
    VALID_MP4 = (b"ftyp",b"free",b"mdat",b"moov",b"wide",b"skip",b"pnot",b"uuid",b"isom",b"mp41",b"mp42",b"avc1",b"MSNV",b"M4V ",b"M4A ",b"f4v ",b"qt  ")

    try:
        with open(tmp_path,"wb") as f:
            async for chunk in request.stream():
                if not chunk: continue
                if first_chunk:
                    first_chunk = False
                    is_video = (
                        (len(chunk)>=12 and chunk[4:8] in VALID_MP4)
                        or (len(chunk)>=4 and chunk[:4] in (b"\x1aE\xdf\xa3",b"RIFF",b"OggS"))
                        or (len(chunk)>=3 and chunk[:3]==b"FLV")
                        or (len(chunk)>=4 and chunk[:4] in (b"\x00\x00\x01\xba",b"\x00\x00\x01\xb3"))
                    )
                    if not is_video:
                        ct = request.headers.get("content-type","")
                        # Accepte si: magic bytes OK OU content-type video/x OU octet-stream OU vide
                        # ffprobe fera la vraie vérification post-upload
                        if not (ct.startswith("video/") or ct.startswith("application/") or ct == ""):
                            raise HTTPException(400,
                                "Ce fichier ne semble pas être une vidéo. "                                "Formats acceptés : MP4, MOV, MKV, AVI, WEBM.")
                f.write(chunk)
                written += len(chunk)
                if written > MAX_FILE_SIZE:
                    raise HTTPException(413,f"Fichier trop lourd (max 2 GB). Ton fichier dépasse cette limite.")
    except HTTPException:
        try: os.remove(tmp_path)
        except: pass
        raise

    if written < 1000:
        try: os.remove(tmp_path)
        except: pass
        raise HTTPException(400,"Fichier vide.")

    try:
        verify_dur = await asyncio.get_running_loop().run_in_executor(None, _get_duration, tmp_path)
    except Exception as ve:
        try: os.remove(tmp_path)
        except: pass
        raise HTTPException(400, f"Fichier non lisible : {ve}")

    jobs[job_id] = {"status":"queued","progress":2,"message":"⬆️ Reçu, analyse en cours...",
                    "clips":[],"created_at":time.time(),"video_title":"","zip_url":None}
    settings = {
        "watermark":       request.headers.get("x-watermark",""),
        "_known_duration": verify_dur,
    }
    background_tasks.add_task(process_video_job, job_id, tmp_path, filename, settings)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str, response: Response):
    if job_id not in jobs: raise HTTPException(404,"Job introuvable.")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return jobs[job_id]

@app.get("/api/jobs")
async def list_jobs():
    result = [{"job_id":jid,"clips":j["clips"],"created_at":j.get("created_at",0)}
              for jid,j in jobs.items() if j.get("status")=="done" and j.get("clips")]
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result[:20]

@app.on_event("startup")
async def start_cleanup():
    asyncio.create_task(_cleanup_loop())

async def _cleanup_loop():
    while True:
        await asyncio.sleep(1800)
        now = time.time()
        for jid in list(jobs.keys()):
            j = jobs.get(jid,{})
            age = now - j.get("created_at",now)
            if age > 10800 or (j.get("status")=="error" and age > 1800):
                job_dir = OUTPUT_DIR / jid
                if job_dir.exists(): shutil.rmtree(job_dir, ignore_errors=True)
                jobs.pop(jid, None)
        for ip in list(ip_jobs.keys()):
            ip_jobs[ip] = [t for t in ip_jobs[ip] if now-t < 3600]
            if not ip_jobs[ip]: del ip_jobs[ip]

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/",        StaticFiles(directory="static", html=True), name="static")
