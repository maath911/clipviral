import os
import gc
import glob
import uuid
import shutil
import subprocess
import threading
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ClipViral API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────
MAX_VIDEO_DURATION = 3600   # 60 min
MAX_FILE_SIZE      = 3 * 1024 * 1024 * 1024  # 3 GB
CLIP_MIN  = 35   # secondes min par clip
CLIP_MAX  = 90   # secondes max par clip
CLIP_GAP  = 60   # gap minimum entre deux clips

# ── In-memory state ─────────────────────────────────────────
jobs: dict              = {}
ws_connections: dict    = defaultdict(list)
ip_jobs: dict           = defaultdict(list)
MAX_JOBS_PER_IP         = 5
executor                = ThreadPoolExecutor(max_workers=1)  # 1 seul job à la fois sur Render free

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def set_job(job_id: str, **kw):
    if job_id in jobs:
        jobs[job_id].update(kw)

async def notify_ws(job_id: str):
    try:
        data = json.dumps(jobs.get(job_id, {}), default=str)
    except Exception:
        try:
            j = jobs.get(job_id, {})
            data = json.dumps({"status": j.get("status", "unknown"),
                               "progress": j.get("progress", 0),
                               "message": "Mise à jour..."})
        except Exception:
            return
    dead = []
    for ws in list(ws_connections.get(job_id, [])):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for d in dead:
        try:
            ws_connections[job_id].remove(d)
        except ValueError:
            pass

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name).strip('_.')
    return name[:80] or "video"

# ─────────────────────────────────────────────────────────────
# 1. DURÉE VIDÉO — robuste pour tous les formats
# ─────────────────────────────────────────────────────────────
def _get_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(video_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise ValueError("Lecture du fichier trop lente. Réessaie.")
    except Exception as e:
        raise ValueError(f"ffprobe indisponible: {e}")

    if not r.stdout or not r.stdout.strip():
        raise ValueError("Fichier vidéo invalide ou corrompu.")
    try:
        info = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise ValueError("Fichier non reconnu. Vérifie que c'est bien une vidéo.")

    # Cherche duration dans format d'abord, puis dans les streams (MKV/AVI/TS n'ont pas toujours la durée dans format)
    dur = None
    raw = info.get("format", {}).get("duration", "N/A")
    if raw not in ("N/A", "", None):
        try:
            d = float(raw)
            if d > 0:
                dur = d
        except (ValueError, TypeError):
            pass

    if not dur:
        for stream in info.get("streams", []):
            raw = stream.get("duration", "N/A")
            if raw not in ("N/A", "", None):
                try:
                    d = float(raw)
                    if d > 0:
                        dur = d
                        break
                except (ValueError, TypeError):
                    pass

    if not dur or dur <= 0:
        raise ValueError("Impossible de lire la durée. Fichier peut-être corrompu ou tronqué.")
    return dur

# ─────────────────────────────────────────────────────────────
# 2. VÉRIFICATION AUDIO — calculé une fois, passé partout
# ─────────────────────────────────────────────────────────────
def _has_audio(video_path: str) -> bool:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", str(video_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        info = json.loads(r.stdout)
        return any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    except Exception:
        return True  # doute → on suppose qu'il y a de l'audio

# ─────────────────────────────────────────────────────────────
# 3. PRÉ-ENCODAGE — H264 garanti pour filter_complex
# ─────────────────────────────────────────────────────────────
def _ensure_h264(video_path: str, job_id: str, known_duration: float = None) -> str:
    # Détection codec
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        streams = json.loads(r.stdout).get("streams", [])
        codec = next((s["codec_name"] for s in streams if s.get("codec_type") == "video"), "h264")
    except subprocess.TimeoutExpired:
        return video_path
    except Exception:
        codec = "h264"

    NEEDS_REENCODE = {"hevc", "h265", "vp9", "av1", "vp8", "wmv3", "mpeg4", "theora"}
    if codec.lower() not in NEEDS_REENCODE:
        return video_path

    # Limite : vidéos longues en codec exotique = trop lent à ré-encoder sur Render free
    dur = known_duration
    if not dur:
        try:
            dur = _get_duration(video_path)
        except Exception:
            dur = 0
    if dur > 1200:  # > 20min
        raise ValueError(
            f"Ta vidéo utilise le codec {codec.upper()} et dure {int(dur//60)}min. "
            f"Ré-encoder prendrait trop de temps sur notre serveur. "
            f"Convertis-la en MP4 H264 avec HandBrake (gratuit) avant de l'uploader."
        )

    out_path = str(Path(video_path).parent / f"{job_id}_h264.mp4")
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", out_path
        ], capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        return video_path  # trop lent → tente avec l'original

    if result.returncode == 0:
        try:
            os.remove(video_path)
        except Exception:
            pass
        return out_path
    return video_path

# ─────────────────────────────────────────────────────────────
# 4. ÉNERGIE AUDIO — par chunks 5min, jamais > 50MB RAM
# ─────────────────────────────────────────────────────────────
def _extract_audio_energy(video_path: str, seg_dur: float = 3.0, duration: float = None):
    if not duration:
        duration = _get_duration(video_path)

    CHUNK = 300  # 5 min
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
                "ffmpeg", "-y",
                "-ss", str(chunk_start), "-i", str(video_path),
                "-t", str(chunk_dur),
                "-vn", "-ac", "1", "-ar", "22050",
                "-f", "f32le", "-"
            ], capture_output=True, timeout=120)
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
            seg = samples[i:i + hop]
            rms      = float(np.sqrt(np.mean(seg ** 2)))
            variance = float(np.var(seg))
            all_times.append(chunk_start + i / sr)
            all_energies.append(rms)
            all_emotions.append(min(1.0, variance * 50))

        del samples

    if not all_times:
        all_times, all_energies, all_emotions = [0.0], [0.3], [0.3]

    return np.array(all_times), np.array(all_energies), np.array(all_emotions), float(duration)

# ─────────────────────────────────────────────────────────────
# 5. WHISPER — timeout strict, jamais bloquant
# ─────────────────────────────────────────────────────────────
def _transcribe_full(video_path: str, duration: float, job_id: str, lang: str = "auto") -> list:
    try:
        import whisper as _whisper
    except ImportError:
        return []

    # Paramètres selon durée — budget total max 7min sur Render free
    if duration <= 600:       interval, max_s = 20, 5
    elif duration <= 1800:    interval, max_s = 40, 5
    elif duration <= 3600:    interval, max_s = 90, 4
    else:                     interval, max_s = 180, 3

    SEG_TIMEOUT    = 90   # s max par segment Whisper
    GLOBAL_TIMEOUT = 420  # 7min max au total
    SAMPLE_LEN     = 28   # s d'audio par segment

    zones = []
    t = 0.0
    while t < duration and len(zones) < max_s:
        zones.append(t)
        t += interval

    try:
        model = _whisper.load_model("tiny")
    except Exception:
        return []

    whisper_lang = None if lang == "auto" else lang
    all_segs     = []
    t_start      = time.monotonic()

    try:
        for zs in zones:
            if time.monotonic() - t_start > GLOBAL_TIMEOUT:
                break  # retourne ce qui est déjà collecté

            ze  = min(zs + SAMPLE_LEN, duration)
            tmp = str(UPLOAD_DIR / f"{job_id}_w_{int(zs)}.wav")

            # Extraction WAV
            try:
                r = subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", str(zs), "-i", str(video_path),
                    "-t", str(ze - zs),
                    "-vn", "-ac", "1", "-ar", "16000",
                    "-f", "wav", tmp
                ], capture_output=True, timeout=45)
            except subprocess.TimeoutExpired:
                try: os.remove(tmp)
                except Exception: pass
                continue

            if r.returncode != 0 or not Path(tmp).exists():
                continue

            # Transcription avec timeout strict via thread daemon
            try:
                holder = [None]
                def _run(tmp_path=tmp, h=holder):
                    try:
                        h[0] = model.transcribe(
                            tmp_path, language=whisper_lang, fp16=False,
                            verbose=False, condition_on_previous_text=False,
                            temperature=0, beam_size=1, best_of=1,
                        )
                    except Exception:
                        pass

                thr = threading.Thread(target=_run, daemon=True)
                thr.start()
                thr.join(timeout=SEG_TIMEOUT)

                res = holder[0] or {"segments": []}
                for seg in res.get("segments", []):
                    text = seg.get("text", "").strip()
                    if text:
                        all_segs.append({
                            "start": round(float(seg["start"]) + zs, 1),
                            "end":   round(float(seg["end"])   + zs, 1),
                            "text":  text,
                        })
            except Exception:
                pass
            finally:
                try: os.remove(tmp)
                except Exception: pass

    finally:
        del model
        gc.collect()

    return all_segs

# ─────────────────────────────────────────────────────────────
# 6. DÉTECTION CLIPS — seuil adaptatif, jamais de crash
# ─────────────────────────────────────────────────────────────
def _auto_detect_clips(times, energies, emotions, segments, duration) -> list:
    # Score combiné
    norm_e  = energies / (energies.max() + 1e-9)
    norm_em = emotions / (emotions.max() + 1e-9)

    VIRAL_WORDS = [
        "incroyable","jamais","secret","vérité","choc","révélation","attention",
        "important","fou","impossible","erreur","problème","argent","millions",
        "gratuit","interdit","caché","voici",
        "shocking","never","truth","crazy","millions","money","free","hidden","banned","watch",
    ]
    word_boost = np.zeros(len(times))
    for seg in segments:
        txt = seg.get("text", "").lower()
        boost = min(0.4, sum(0.08 for w in VIRAL_WORDS if w in txt))
        if boost > 0:
            mask = (times >= seg["start"]) & (times <= seg["end"])
            word_boost[mask] += boost

    combined = norm_e * 0.60 + norm_em * 0.25 + np.clip(word_boost, 0, 0.4) * 0.15
    window   = max(3, int(10 / 3))
    smoothed = np.convolve(combined, np.ones(window) / window, mode="same")

    mean_s = smoothed.mean()
    std_s  = smoothed.std()
    cv     = std_s / (mean_s + 1e-6)
    k      = 0.8 if cv > 0.5 else (1.0 if cv > 0.3 else 0.5)
    threshold = mean_s + k * std_s

    if   duration <= 900:  max_clips = 3
    elif duration <= 1800: max_clips = 5
    elif duration <= 3600: max_clips = 8
    else:                  max_clips = 10

    # Pics locaux
    peaks = [i for i in range(1, len(smoothed) - 1)
             if smoothed[i] >= threshold
             and smoothed[i] >= smoothed[i-1]
             and smoothed[i] >= smoothed[i+1]]
    peaks.sort(key=lambda i: smoothed[i], reverse=True)

    ext_thresh = mean_s + 0.2 * std_s
    clips = []

    for pi in peaks:
        if len(clips) >= max_clips:
            break
        peak_t = float(times[pi])

        # Pas de chevauchement
        if any(abs(peak_t - (c["start"] + c["end"]) / 2) < CLIP_GAP for c in clips):
            continue

        # Extension gauche
        left = pi
        while left > 0 and (peak_t - times[left-1]) <= CLIP_MAX and smoothed[left-1] >= ext_thresh:
            left -= 1

        # Extension droite
        right = pi
        while right < len(times) - 1:
            if (times[right+1] - times[left]) > CLIP_MAX:
                break
            if smoothed[right+1] >= ext_thresh:
                right += 1
            elif (times[right] - times[left]) < CLIP_MIN:
                right += 1
            else:
                break

        s = float(times[left])
        e = float(times[right])

        # Durée min
        if (e - s) < CLIP_MIN:
            e = min(s + CLIP_MIN, duration)
        # Durée max
        if (e - s) > CLIP_MAX:
            half = CLIP_MAX / 2
            s = max(0.0, peak_t - half)
            e = min(duration, s + CLIP_MAX)
        # Borne fin
        if e > duration:
            e = duration
            s = max(0.0, e - CLIP_MIN)

        # Sécurité : durée > 0
        if e - s <= 0:
            continue

        mask = (times >= s) & (times <= e)
        vs   = min(1.0, float(smoothed[mask].mean()) / (smoothed.max() + 1e-6)) if mask.any() else 0.5

        clips.append({"start": round(s, 1), "end": round(e, 1),
                      "duration": round(e - s, 1), "viral_score": round(vs, 3), "reason": ""})

    # Fallback si aucun pic
    if not clips:
        n = min(3, max(1, int(duration / 600)))
        step = duration / (n + 1)
        for i in range(n):
            s = max(0.0, round(step * (i + 1) - CLIP_MIN / 2, 1))
            e = min(float(duration), s + CLIP_MIN)
            if e - s < 5:
                s = max(0.0, float(duration) - 10)
                e = float(duration)
            if e - s > 0:
                clips.append({"start": round(s, 1), "end": round(e, 1),
                              "duration": round(e - s, 1), "viral_score": 0.5, "reason": "Fallback"})

    clips.sort(key=lambda x: x["start"])
    return clips

# ─────────────────────────────────────────────────────────────
# 7. SCORE TIKTOK A→F
# ─────────────────────────────────────────────────────────────
def _tiktok_score(clip: dict, segments: list, duration: float) -> dict:
    s, e  = clip["start"], clip["end"]
    dur   = e - s
    base  = clip.get("viral_score", 0.5)
    total = 0.0
    det   = {}

    # Durée
    if 45 <= dur <= 90:   ds, dt = 1.0, "✅ Durée parfaite pour TikTok (45-90s)"
    elif 30 <= dur < 45:  ds, dt = 0.8, "🟡 Un peu court — idéal = 45-90s"
    elif 90 < dur <= 120: ds, dt = 0.8, "🟡 Un peu long — idéal = 45-90s"
    elif dur < 30:        ds, dt = 0.5, "⚠️ Trop court pour TikTok"
    else:                 ds, dt = 0.6, "⚠️ Trop long — découpe en 2"
    det["duree"] = {"score": ds, "tip": dt}
    total += ds * 20

    # Accroche
    hook_txt   = " ".join(sg["text"] for sg in segments if s <= sg["start"] <= s + 8).lower()
    HOOK_WORDS = ["attention","regarde","jamais","secret","incroyable","vérité","choc",
                  "wait","stop","wow","shocking","never","hidden","watch","fou","impossible"]
    hit = sum(1 for w in HOOK_WORDS if w in hook_txt)
    if hit >= 2:   hs, ht = 1.0, "✅ Accroche forte dès le début"
    elif hit == 1: hs, ht = 0.7, "🟡 Accroche moyenne — commence par un mot fort"
    else:          hs, ht = 0.4, "❌ Accroche faible — mets un texte choc en intro"
    det["accroche"] = {"score": hs, "tip": ht}
    total += hs * 25

    # Densité parole
    clip_segs  = [sg for sg in segments if s <= sg["start"] <= e]
    words_min  = len(" ".join(sg["text"] for sg in clip_segs).split()) / max(dur / 60, 0.1)
    if words_min >= 100:   ss, st = 1.0,  "✅ Rythme de parole parfait"
    elif words_min >= 60:  ss, st = 0.75, "🟡 Rythme correct"
    elif words_min >= 25:  ss, st = 0.5,  "⚠️ Peu de paroles"
    else:                  ss, st = 0.3,  "❌ Quasi pas de paroles — ajoute une voix off ?"
    det["parole"] = {"score": ss, "tip": st}
    total += ss * 20

    # Engagement
    full_txt = " ".join(sg["text"] for sg in clip_segs)
    eng = min(1.0, full_txt.count("?") * 0.12 + full_txt.count("!") * 0.08 + base * 0.5)
    det["engagement"] = {"score": eng,
                         "tip": "✅ Contenu engageant" if eng >= 0.7
                                else "🟡 Ajoute des questions pour les commentaires"}
    total += eng * 20
    det["energie"] = {"score": base,
                      "tip": "✅ Énergie forte" if base >= 0.7 else "🟡 Énergie correcte"}
    total += base * 15

    pct = round(total)
    if   pct >= 85: grade, gc = "A", "#00dfa2"
    elif pct >= 70: grade, gc = "B", "#5b8dee"
    elif pct >= 55: grade, gc = "C", "#f5c842"
    elif pct >= 40: grade, gc = "D", "#ff8c00"
    else:           grade, gc = "F", "#ff2d55"
    return {"grade": grade, "grade_color": gc, "tiktok_score": pct, "details": det}

# ─────────────────────────────────────────────────────────────
# 8. SOUS-TITRES SRT
# ─────────────────────────────────────────────────────────────
def _generate_srt(segments: list, clip_start: float, clip_end: float, out_srt: str) -> bool:
    try:
        segs = [sg for sg in segments
                if sg["start"] >= clip_start - 0.5 and sg["end"] <= clip_end + 0.5]
        if not segs:
            return False

        def ts(sec):
            sec = max(0.0, float(sec))
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

        lines = []
        for i, sg in enumerate(segs, 1):
            s = max(0.0, sg["start"] - clip_start)
            e = min(clip_end - clip_start, sg["end"] - clip_start)
            if e <= s:
                continue
            txt = sg["text"].strip()
            if not txt:
                continue
            if len(txt) > 42:
                mid = len(txt) // 2
                sp  = txt.rfind(" ", 0, mid)
                if sp > 0:
                    txt = txt[:sp] + "\n" + txt[sp+1:]
            lines.append(f"{i}\n{ts(s)} --> {ts(e)}\n{txt}\n")

        if not lines:
            return False
        with open(out_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────
# 9. EXPORT TIKTOK 1080×1920
# ─────────────────────────────────────────────────────────────
def _export_clip(input_path: str, start: float, end: float, out_path: str,
                 srt_path: str = None, watermark: str = None,
                 has_audio: bool = True) -> bool:
    dur  = end - start
    if dur <= 0:
        return False
    fade = max(0.0, dur - 2.0)

    filters = [
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,boxblur=20:20,eq=brightness=-0.4[bg]",
        "[0:v]scale=1080:-2[fg]",
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fade=t=out:st={fade}:d=2[v0]",
    ]
    current = "[v0]"
    idx     = 1

    if srt_path and Path(srt_path).exists():
        safe = str(srt_path).replace("\\", "/").replace("'", "\\'").replace(":", "\\:")
        nxt  = f"[v{idx}]"
        filters.append(
            f"{current}subtitles='{safe}':force_style="
            f"'FontName=Arial,FontSize=14,Bold=1,"
            f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
            f"Outline=2,Shadow=1,Alignment=2,MarginV=60'{nxt}"
        )
        current = nxt
        idx += 1

    if watermark:
        safe_wm = (watermark
                   .replace("\\", "\\\\")
                   .replace("'",  "\\'")
                   .replace(":",  "\\:")
                   .replace("%",  "\\%"))
        nxt = f"[v{idx}]"
        filters.append(
            f"{current}drawtext=text='{safe_wm}':"
            f"fontsize=26:fontcolor=white@0.65:x=18:y=18{nxt}"
        )
        current = nxt

    vf  = ";".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", str(input_path),
        "-t", str(dur),
        "-filter_complex", vf,
        "-map", current,
    ]
    if has_audio:
        # 0:a? = optionnel — si le stream audio disparaît entre has_audio check et ici, pas de crash
        cmd += ["-map", "0:a?", "-af", f"afade=t=out:st={fade}:d=2", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────
# 10. CAPTION
# ─────────────────────────────────────────────────────────────
def _make_caption(clip: dict) -> str:
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
# 11. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────
async def process_video_job(job_id: str, video_path: str, filename: str,
                            settings: dict = None):
    if settings is None:
        settings = {}
    job_dir   = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    loop      = asyncio.get_running_loop()
    lang      = settings.get("lang", "auto")
    subtitles = settings.get("subtitles", True)
    watermark = settings.get("watermark", "").strip()[:40]

    async def upd(pct: int, msg: str, **kw):
        set_job(job_id, progress=pct, message=msg, **kw)
        await notify_ws(job_id)

    try:
        # ── 1. Vérification ─────────────────────────────────
        size_mb = Path(video_path).stat().st_size // (1024 * 1024)
        title   = Path(filename).stem[:60]
        set_job(job_id, video_title=title)
        await upd(5, f"✅ Vidéo reçue ({size_mb} MB) — vérification...")

        duration = settings.get("_known_duration") or \
                   await loop.run_in_executor(executor, _get_duration, video_path)
        duration = float(duration)

        if duration > MAX_VIDEO_DURATION:
            raise ValueError(
                f"Vidéo trop longue ({int(duration//60)}min). "
                f"Maximum 60 min. Découpe avec VLC puis réenvoie."
            )
        mins = int(duration // 60)
        secs = int(duration % 60)
        await upd(8, f"⏱️ Durée : {mins}min {secs}s — vérification codec...")

        # ── 2. Codec ────────────────────────────────────────
        video_path = await loop.run_in_executor(
            executor, _ensure_h264, video_path, job_id, duration)
        await upd(12, "🔧 Format OK. Analyse audio en cours...")

        # ── 3. Audio energy ─────────────────────────────────
        times, energies, emotions, duration = await loop.run_in_executor(
            executor, _extract_audio_energy, video_path, 3.0, duration)
        await upd(28, f"🎵 Audio analysé ({mins}min). Transcription Whisper...")

        # ── 4. Whisper ──────────────────────────────────────
        await upd(30, f"✍️ Transcription Whisper en cours — max 7min, patiente...")
        segments = await loop.run_in_executor(
            executor, _transcribe_full, video_path, duration, job_id, lang)
        n_seg = len(segments)
        await upd(58,
            (f"✍️ {n_seg} segments transcrits." if n_seg > 0
             else "✍️ Transcription terminée (peu de paroles détectées).")
            + " Détection des moments forts...")

        # ── 5. Détection ────────────────────────────────────
        clips = await loop.run_in_executor(
            executor, _auto_detect_clips, times, energies, emotions, segments, duration)
        if not clips:
            raise ValueError(
                "Aucun moment fort détecté. Essaie avec une vidéo avec plus de paroles ou de musique.")
        await upd(65,
            f"✅ {len(clips)} moment(s) fort(s) détecté(s). Export TikTok...",
            status="exporting")

        # ── 6. Export ───────────────────────────────────────
        has_audio = await loop.run_in_executor(executor, _has_audio, video_path)
        exported  = []

        for i, clip in enumerate(clips):
            name     = f"Clip_Elite_{i+1}"
            out_path = str(job_dir / f"{name}.mp4")
            srt_path = str(job_dir / f"{name}.srt") if subtitles else None

            if subtitles and srt_path:
                await loop.run_in_executor(
                    executor, _generate_srt, segments, clip["start"], clip["end"], srt_path)

            ok = await loop.run_in_executor(
                executor, _export_clip,
                video_path, clip["start"], clip["end"], out_path,
                srt_path, watermark or None, has_audio)

            if ok:
                score = _tiktok_score(clip, segments, duration)
                exported.append({
                    **clip,
                    "filename":      f"{name}.mp4",
                    "url":           f"/outputs/{job_id}/{name}.mp4",
                    "preview_url":   f"/outputs/{job_id}/{name}.mp4",
                    "rank":          i + 1,
                    "caption":       _make_caption(clip),
                    "video_title":   title,
                    "tiktok_score":  score["tiktok_score"],
                    "grade":         score["grade"],
                    "grade_color":   score["grade_color"],
                    "tt_details":    score["details"],
                    "has_subtitles": subtitles,
                })
            pct = 65 + int((i + 1) / len(clips) * 32)
            await upd(pct, f"🎬 Export {i+1}/{len(clips)} ({int(clip['duration'])}s)...")

        if not exported:
            raise ValueError("Tous les clips ont échoué à l'export. Réessaie avec une autre vidéo.")

        # ── 7. ZIP ──────────────────────────────────────────
        zip_url = None
        zip_path = job_dir / "tous_les_clips.zip"
        def _make_zip():
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                for c in exported:
                    p = job_dir / c["filename"]
                    if p.exists():
                        zf.write(p, c["filename"])
        await loop.run_in_executor(executor, _make_zip)
        if zip_path.exists():
            zip_url = f"/outputs/{job_id}/tous_les_clips.zip"

        set_job(job_id, status="done", progress=100,
                clips=exported, zip_url=zip_url,
                message=f"🎉 {len(exported)} clips viraux prêts !")
        await notify_ws(job_id)

    except Exception as e:
        set_job(job_id, status="error", message=f"❌ {str(e)}")
        await notify_ws(job_id)

    finally:
        # Nettoyage systématique de TOUS les fichiers temporaires du job
        for p in [
            video_path,
            str(UPLOAD_DIR / f"{job_id}.mp4"),
            str(UPLOAD_DIR / f"{job_id}_h264.mp4"),
        ]:
            try:
                if Path(p).exists():
                    os.remove(p)
            except Exception:
                pass
        for wav in glob.glob(str(UPLOAD_DIR / f"{job_id}_w_*.wav")):
            try:
                os.remove(wav)
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    ws_connections[job_id].append(websocket)
    try:
        if job_id in jobs:
            await websocket.send_text(json.dumps(jobs[job_id], default=str))
        while True:
            await asyncio.sleep(15)
            try:
                await websocket.send_text('{"ping":true}')
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            ws_connections[job_id].remove(websocket)
        except ValueError:
            pass

# ─────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, request: Request):
    # Anti-abus par IP
    client_ip = request.client.host if request.client else "unknown"
    now       = time.time()
    ip_jobs[client_ip] = [t for t in ip_jobs[client_ip] if now - t < 3600]
    if len(ip_jobs[client_ip]) >= MAX_JOBS_PER_IP:
        raise HTTPException(429, "Trop de vidéos envoyées. Attends 1h avant de réessayer.")
    ip_jobs[client_ip].append(now)

    job_id   = str(uuid.uuid4())[:8]
    tmp_path = str(UPLOAD_DIR / f"{job_id}.mp4")

    # Nom de fichier depuis Content-Disposition
    cd = request.headers.get("content-disposition", "")
    filename = "video.mp4"
    if "filename=" in cd:
        try:
            raw = cd.split("filename=")[1].strip().strip('"\'')
            filename = sanitize_filename(raw) + ".mp4"
        except Exception:
            pass

    # ── Streaming vers disque ────────────────────────────────
    written     = 0
    first_chunk = True
    VALID_MP4   = (b"ftyp", b"free", b"mdat", b"moov", b"wide")

    try:
        with open(tmp_path, "wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue

                # Vérification magic bytes sur le 1er chunk uniquement
                if first_chunk:
                    first_chunk = False
                    is_video = (
                        (len(chunk) >= 12 and chunk[4:8] in VALID_MP4)
                        or (len(chunk) >= 4 and chunk[:4] in (b"\x1aE\xdf\xa3", b"RIFF", b"OggS"))
                        or (len(chunk) >= 3 and chunk[:3] == b"FLV")
                        or (len(chunk) >= 4 and chunk[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"))
                    )
                    if not is_video:
                        ct = request.headers.get("content-type", "")
                        if not (ct.startswith("video/") or ct in ("application/octet-stream", "")):
                            raise HTTPException(400, "Ce fichier n'est pas une vidéo reconnue (MP4, MKV, MOV, AVI, WEBM).")

                f.write(chunk)
                written += len(chunk)

                if written > MAX_FILE_SIZE:
                    raise HTTPException(413, "Fichier trop lourd (maximum 3 GB).")

    except HTTPException:
        try: os.remove(tmp_path)
        except Exception: pass
        raise

    if written < 1000:
        try: os.remove(tmp_path)
        except Exception: pass
        raise HTTPException(400, "Fichier vide ou trop petit.")

    # Vérification intégrité post-upload (détecte fichiers tronqués, non-vidéo)
    try:
        verify_dur = await asyncio.get_running_loop().run_in_executor(
            None, _get_duration, tmp_path)
    except Exception as ve:
        try: os.remove(tmp_path)
        except Exception: pass
        raise HTTPException(400, f"Fichier non lisible : {ve}")

    # Crée le job
    jobs[job_id] = {
        "status":      "queued",
        "progress":    2,
        "message":     "⬆️ Fichier reçu, démarrage de l'analyse...",
        "clips":       [],
        "created_at":  time.time(),
        "video_title": "",
        "zip_url":     None,
    }
    settings = {
        "lang":             request.headers.get("x-lang", "auto"),
        "subtitles":        request.headers.get("x-subtitles", "true").lower() == "true",
        "watermark":        request.headers.get("x-watermark", ""),
        "_known_duration":  verify_dur,
    }
    background_tasks.add_task(process_video_job, job_id, tmp_path, filename, settings)
    return {"job_id": job_id}

# ─────────────────────────────────────────────────────────────
# STATUS & JOBS
# ─────────────────────────────────────────────────────────────
@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job introuvable.")
    return jobs[job_id]

@app.get("/api/jobs")
async def list_jobs():
    result = [
        {"job_id": jid, "clips": j["clips"], "created_at": j.get("created_at", 0)}
        for jid, j in jobs.items()
        if j.get("status") == "done" and j.get("clips")
    ]
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result[:20]

# ─────────────────────────────────────────────────────────────
# NETTOYAGE AUTO
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def start_cleanup():
    asyncio.create_task(_cleanup_loop())

async def _cleanup_loop():
    while True:
        await asyncio.sleep(1800)  # toutes les 30min
        now = time.time()
        for jid in list(jobs.keys()):
            j = jobs.get(jid, {})
            age = now - j.get("created_at", now)
            status = j.get("status", "")
            # Supprime : jobs > 3h OU jobs en erreur > 30min (libère mémoire)
            if age > 10800 or (status == "error" and age > 1800):
                job_dir = OUTPUT_DIR / jid
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                jobs.pop(jid, None)
        # Purge ip_jobs pour éviter accumulation mémoire illimitée
        for ip in list(ip_jobs.keys()):
            ip_jobs[ip] = [t for t in ip_jobs[ip] if now - t < 3600]
            if not ip_jobs[ip]:
                del ip_jobs[ip]

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/",        StaticFiles(directory="static", html=True), name="static")
