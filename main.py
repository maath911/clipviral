import os
import uuid
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
MAX_VIDEO_DURATION = 3600       # 60 min max
CLIP_MIN = 35                   # secondes min par clip
CLIP_MAX = 90                   # secondes max par clip (idéal TikTok)
CLIP_GAP = 60                   # gap minimum entre deux clips

# ── In-memory state ─────────────────────────────────────────
jobs: dict = {}
ws_connections: dict = defaultdict(list)
ip_jobs: dict = defaultdict(list)
MAX_JOBS_PER_IP = 5
executor = ThreadPoolExecutor(max_workers=2)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def set_job(job_id, **kw):
    if job_id in jobs:
        jobs[job_id].update(kw)

async def notify_ws(job_id):
    try:
        data = json.dumps(jobs.get(job_id, {}), default=str)
    except Exception:
        data = json.dumps({"status": jobs.get(job_id, {}).get("status", "unknown"),
                           "progress": jobs.get(job_id, {}).get("progress", 0),
                           "message": "Mise à jour en cours..."})
    dead = []
    for ws in ws_connections.get(job_id, []):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for d in dead:
        ws_connections[job_id].remove(d)

def sanitize_filename(name: str) -> str:
    """Nettoie le nom de fichier — évite les crashes sur noms spéciaux."""
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name).strip('_.')
    return name[:80] or "video"

# ─────────────────────────────────────────────────────────────
# 1. DURÉE VIDÉO
# ─────────────────────────────────────────────────────────────
def _get_duration(video_path: str) -> float:
    """
    Retourne la durée en secondes.
    Gère: fichiers corrompus, formats sans métadonnée duration (MKV/AVI/TS),
    sortie ffprobe vide, valeurs N/A.
    """
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(video_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if not r.stdout or r.stdout.strip() == "":
            raise ValueError("ffprobe n'a retourné aucune donnée — fichier invalide")
        info = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise ValueError("Fichier invalide ou corrompu. Vérifie que c'est bien une vidéo.")
    except subprocess.TimeoutExpired:
        raise ValueError("Impossible de lire le fichier vidéo (timeout).")

    # Cherche la durée dans format d'abord, puis dans les streams (MKV/AVI/TS)
    dur = None
    fmt_dur = info.get("format", {}).get("duration", "N/A")
    if fmt_dur != "N/A":
        try:
            dur = float(fmt_dur)
        except (ValueError, TypeError):
            pass

    if not dur or dur <= 0:
        # Fallback: cherche dans les streams vidéo/audio
        for stream in info.get("streams", []):
            sdur = stream.get("duration", "N/A")
            if sdur != "N/A":
                try:
                    d = float(sdur)
                    if d > 0:
                        dur = d
                        break
                except (ValueError, TypeError):
                    pass

    if not dur or dur <= 0:
        raise ValueError("Impossible de déterminer la durée. Le fichier est peut-être corrompu.")

    return dur

# ─────────────────────────────────────────────────────────────
# 2. PRÉ-ENCODAGE (compatibilité maximale)
#    Re-encode en H264 si HEVC/VP9/AV1 pour éviter tout crash FFmpeg
# ─────────────────────────────────────────────────────────────
def _ensure_h264(video_path: str, job_id: str, known_duration: float = None) -> str:
    """
    Vérifie le codec vidéo. Si pas H264, re-encode en H264.
    Garantit 100% compatibilité FFmpeg pour tous les cas.
    known_duration: durée déjà calculée pour éviter un double ffprobe.
    """
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", str(video_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        streams = json.loads(r.stdout).get("streams", [])
        codec = next((s["codec_name"] for s in streams if s.get("codec_type") == "video"), "h264")
    except subprocess.TimeoutExpired:
        return video_path  # ffprobe trop lent → on tente avec l'original
    except Exception:
        codec = "h264"

    # Codecs qui nécessitent un re-encode pour compatibilité FFmpeg filter_complex
    needs_reencode = codec.lower() in ("hevc", "h265", "vp9", "av1", "vp8", "wmv3", "mpeg4")

    if not needs_reencode:
        return video_path  # déjà compatible

    out_path = str(Path(video_path).parent / f"{job_id}_h264.mp4")
    cmd_re = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", out_path
    ]
    # Sur Render free: re-encoder une vidéo longue peut dépasser 20min
    # On limite: si durée > 20min ET besoin de re-encoder → on rejette avec message clair
    try:
        d = known_duration if known_duration else _get_duration(video_path)
        if d > 1200:  # > 20min en codec non-H264
            raise ValueError(
                f"Ta vidéo utilise le codec {codec.upper()} et dure {int(d//60)}min. "
                f"Re-encoder prendrait trop de temps. "
                f"Télécharge-la en MP4 H264 360p depuis 4K Video Downloader."
            )
    except ValueError:
        raise
    except Exception:
        pass  # si on peut pas lire la durée, on tente quand même

    try:
        result = subprocess.run(cmd_re, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        # Re-encode trop lent → on tente avec l'original (peut marcher selon le codec)
        return video_path

    if result.returncode == 0:
        try:
            os.remove(video_path)
        except Exception:
            pass
        return out_path
    return video_path  # si échec, on essaie quand même avec l'original

# ─────────────────────────────────────────────────────────────
# 3. EXTRACTION AUDIO ENERGY (par chunks de 5min — anti-RAM)
# ─────────────────────────────────────────────────────────────
def _extract_audio_energy(video_path: str, seg_dur: float = 3.0, duration: float = None):
    """
    Extrait l'énergie RMS audio par segments de 3s.
    Traitement par chunks de 5min pour ne jamais dépasser 50MB RAM.
    Retourne (times, energies, emotions, total_duration).
    """
    if duration is None:
        duration = _get_duration(video_path)
    chunk_size = 300  # 5 minutes par chunk

    all_times = []
    all_energies = []
    all_emotions = []

    for chunk_start in range(0, int(duration), chunk_size):
        chunk_end = min(chunk_start + chunk_size, duration)
        chunk_dur = chunk_end - chunk_start

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(chunk_start), "-i", str(video_path),
            "-t", str(chunk_dur),
            "-vn", "-ac", "1", "-ar", "22050",
            "-f", "f32le", "-"
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            # Chunk trop lent → scores neutres pour cette portion
            n = int(chunk_dur / seg_dur)
            for i in range(n):
                all_times.append(chunk_start + i * seg_dur)
                all_energies.append(0.3)
                all_emotions.append(0.3)
            continue

        if not r.stdout or len(r.stdout) < 200:
            # Pas d'audio dans ce chunk — scores neutres
            n = int(chunk_dur / seg_dur)
            for i in range(n):
                all_times.append(chunk_start + i * seg_dur)
                all_energies.append(0.3)
                all_emotions.append(0.3)
            continue

        try:
            samples = np.frombuffer(r.stdout, dtype=np.float32)
        except Exception:
            # Données audio corrompues → scores neutres pour ce chunk
            n = int(chunk_dur / seg_dur)
            for i in range(n):
                all_times.append(chunk_start + i * seg_dur)
                all_energies.append(0.3)
                all_emotions.append(0.3)
            continue
        if len(samples) < 100:
            # Trop peu de données → scores neutres
            n = int(chunk_dur / seg_dur)
            for i in range(n):
                all_times.append(chunk_start + i * seg_dur)
                all_energies.append(0.3)
                all_emotions.append(0.3)
            continue
        sr = 22050
        hop = int(seg_dur * sr)

        for i in range(0, len(samples) - hop, hop):
            chunk_samples = samples[i:i + hop]
            rms = float(np.sqrt(np.mean(chunk_samples ** 2)))

            # Détection émotions via variance (voix émotionnelle = variance haute)
            variance = float(np.var(chunk_samples))
            emotion_score = min(1.0, variance * 50)

            all_times.append(chunk_start + i / sr)
            all_energies.append(rms)
            all_emotions.append(emotion_score)

        del samples  # Libère RAM immédiatement

    if not all_times:
        all_times = [0.0]
        all_energies = [0.3]
        all_emotions = [0.3]

    return np.array(all_times), np.array(all_energies), np.array(all_emotions), duration

# ─────────────────────────────────────────────────────────────
# 4. WHISPER TRANSCRIPTION (sampling adaptatif)
# ─────────────────────────────────────────────────────────────
def _transcribe_full(video_path: str, duration: float, job_id: str, lang: str = "auto") -> list:
    """
    Transcrit la vidéo avec Whisper tiny.
    Sampling adaptatif selon durée pour rester sous 15min de traitement.
    """
    try:
        import whisper
    except ImportError:
        return []  # Whisper non installé — mode audio seulement

    # Intervalle de sampling selon durée
    if duration <= 1800:      # ≤ 30min → toutes les 25s (overlap 3s)
        interval = 25
    elif duration <= 3600:    # ≤ 60min → toutes les 50s (overlap 3s)
        interval = 50
    else:
        interval = 90

    sample_len = 28  # durée de chaque sample Whisper (secondes)
    # Sur Render free (CPU partagé), chaque transcription peut prendre 2-5min.
    # Budget total Whisper: 12min max → 12min / 2min par sample = 6 samples max.
    # On prend le minimum entre la couverture optimale et la limite sécurisée.
    if duration <= 1800:    # ≤ 30min
        max_samples = 8     # couvre bien la vidéo
    elif duration <= 3600:  # ≤ 60min
        max_samples = 10    # adapté
    else:
        max_samples = 12    # absolu max

    zones = []
    t = 0.0
    while t < duration and len(zones) < max_samples:
        zones.append(t)
        t += interval

    model = whisper.load_model("tiny")
    whisper_lang = None if lang == "auto" else lang
    all_segs = []

    try:
        for zs in zones:
            ze = min(zs + sample_len, duration)
            tmp = str(UPLOAD_DIR / f"{job_id}_w_{int(zs)}.wav")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(zs), "-i", str(video_path),
                "-t", str(ze - zs),
                "-vn", "-ac", "1", "-ar", "16000",
                "-f", "wav", tmp
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                # Extraction WAV trop lente → skip ce segment
                try: os.remove(tmp)
                except: pass
                continue
            if r.returncode != 0 or not Path(tmp).exists():
                continue
            try:
                # Timeout de 5min par segment via threading pour éviter blocage infini
                import threading
                result_holder = [None]
                def _run_transcribe():
                    result_holder[0] = model.transcribe(
                        tmp, language=whisper_lang, fp16=False,
                        verbose=False, condition_on_previous_text=False)
                t = threading.Thread(target=_run_transcribe, daemon=True)
                t.start()
                t.join(timeout=300)  # 5min max par segment
                res = result_holder[0] if result_holder[0] is not None else {"segments": []}
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
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    finally:
        # Libère TOUJOURS le modèle Whisper (~150MB RAM) même si exception
        del model
        import gc; gc.collect()

    return all_segs

# ─────────────────────────────────────────────────────────────
# 5. DÉTECTION AUTOMATIQUE INTELLIGENTE DES CLIPS
#    Pas de nombre fixe — détecte TOUS les vrais pics
#    Min clips = 1, max selon durée vidéo, seuil adaptatif
# ─────────────────────────────────────────────────────────────
def _auto_detect_clips(times, energies, emotions, segments, duration):
    """
    Détecte automatiquement tous les vrais moments forts.
    Seuil adaptatif = mean + k*std (k ajusté selon la densité).
    Pas de nombre fixe : on prend tous les vrais pics, ni plus ni moins.
    """
    # Score combiné : 60% énergie audio + 25% émotions + 15% mots viraux
    if energies.max() > 0:
        norm_e = energies / energies.max()
    else:
        norm_e = np.ones_like(energies) * 0.3

    if emotions.max() > 0:
        norm_em = emotions / emotions.max()
    else:
        norm_em = np.ones_like(emotions) * 0.3

    # Boost sur les zones avec mots viraux dans la transcription
    viral_words = [
        # FR
        "incroyable", "jamais", "secret", "vérité", "choc", "révélation",
        "attention", "important", "fou", "impossible", "erreur", "problème",
        "argent", "millions", "gratuit", "interdit", "caché", "voici",
        # EN
        "shocking", "never", "secret", "truth", "crazy", "impossible",
        "millions", "money", "free", "hidden", "banned", "watch",
    ]
    word_boost = np.zeros(len(times))
    for seg in segments:
        text_lower = seg["text"].lower()
        boost = sum(0.08 for w in viral_words if w in text_lower)
        boost = min(boost, 0.4)
        if boost > 0:
            mask = (times >= seg["start"]) & (times <= seg["end"])
            word_boost[mask] += boost

    combined = norm_e * 0.60 + norm_em * 0.25 + np.clip(word_boost, 0, 0.4) * 0.15

    # Lissage
    window = max(3, int(10 / 3))  # ~10s de lissage avec segments de 3s
    smoothed = np.convolve(combined, np.ones(window) / window, mode="same")

    # Seuil adaptatif
    mean_s = smoothed.mean()
    std_s  = smoothed.std()

    # k adaptatif : plus strict si beaucoup de variations
    cv = std_s / (mean_s + 1e-6)  # coefficient de variation
    if cv > 0.5:
        k = 0.8   # signal très varié → seuil moins strict
    elif cv > 0.3:
        k = 1.0   # normal
    else:
        k = 0.5   # signal plat → seuil plus bas pour quand même trouver des clips

    peak_threshold = mean_s + k * std_s

    # Limite max clips selon durée
    if duration <= 900:       # ≤ 15min
        max_clips = 3
    elif duration <= 1800:    # ≤ 30min
        max_clips = 5
    elif duration <= 3600:    # ≤ 60min
        max_clips = 8
    else:
        max_clips = 10

    # Détection des pics locaux
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= peak_threshold:
            if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                peaks.append(i)

    # Tri par score décroissant
    peaks.sort(key=lambda i: smoothed[i], reverse=True)

    clips = []
    for peak_idx in peaks:
        if len(clips) >= max_clips:
            break

        peak_time = times[peak_idx]

        # Évite le chevauchement avec un clip existant
        if any(abs(peak_time - ((s + e) / 2)) < CLIP_GAP for s, e in
               [(c["start"], c["end"]) for c in clips]):
            continue

        # Extension gauche
        left = peak_idx
        extend_threshold = mean_s + 0.2 * std_s
        while left > 0:
            if peak_time - times[left - 1] > CLIP_MAX:
                break
            if smoothed[left - 1] >= extend_threshold:
                left -= 1
            else:
                break

        # Extension droite
        right = peak_idx
        while right < len(times) - 1:
            clip_dur = times[right + 1] - times[left]
            if clip_dur > CLIP_MAX:
                break
            if smoothed[right + 1] >= extend_threshold:
                right += 1
            else:
                if times[right] - times[left] < CLIP_MIN:
                    right += 1  # force la durée minimum
                else:
                    break

        start_t = float(times[left])
        end_t   = float(times[right])
        dur_c   = end_t - start_t

        # Ajustement durée minimum
        if dur_c < CLIP_MIN:
            end_t = min(start_t + CLIP_MIN, duration)
            dur_c = end_t - start_t

        # Ajustement durée maximum
        if dur_c > CLIP_MAX:
            # Recentre sur le pic
            half = CLIP_MAX / 2
            start_t = max(0, peak_time - half)
            end_t   = min(duration, start_t + CLIP_MAX)

        # Ne dépasse pas la fin de la vidéo
        if end_t > duration:
            end_t   = duration
            start_t = max(0, end_t - CLIP_MIN)

        mask = (times >= start_t) & (times <= end_t)
        viral_score = float(smoothed[mask].mean()) if mask.any() else 0.5

        # Normalise le score viral entre 0 et 1 par rapport au max du signal
        viral_score = min(1.0, viral_score / (smoothed.max() + 1e-6))

        clips.append({
            "start":       round(start_t, 1),
            "end":         round(end_t, 1),
            "duration":    round(end_t - start_t, 1),
            "viral_score": round(viral_score, 3),
            "reason":      "",
        })

    # Si aucun pic détecté → fallback : prend les N zones les plus énergétiques
    if not clips:
        n_fallback = min(3, max(1, int(duration / 600)))
        step = duration / (n_fallback + 1)
        for i in range(n_fallback):
            s = round(step * (i + 1) - CLIP_MIN / 2, 1)
            s = max(0.0, s)
            # Garantit que le clip ne dépasse pas la durée réelle
            e = min(float(duration), s + CLIP_MIN)
            # Si la vidéo est trop courte pour CLIP_MIN, prend tout ce qui reste
            if e - s < 5:
                s = max(0.0, float(duration) - 10)
                e = float(duration)
            clips.append({
                "start": round(s, 1), "end": round(e, 1),
                "duration": round(e - s, 1),
                "viral_score": 0.5, "reason": "Fallback automatique",
            })

    # Tri final par position chronologique
    clips.sort(key=lambda x: x["start"])
    return clips

# ─────────────────────────────────────────────────────────────
# 6. SCORE TIKTOK A→F (100% gratuit, sans Claude API)
# ─────────────────────────────────────────────────────────────
def _tiktok_score(clip, segments, duration) -> dict:
    start, end = clip["start"], clip["end"]
    dur = end - start
    base = clip.get("viral_score", 0.5)
    total = 0.0
    details = {}

    # Durée optimale (TikTok préfère 45-90s)
    if 45 <= dur <= 90:
        ds = 1.0; dt = "✅ Durée parfaite pour TikTok (45-90s)"
    elif 30 <= dur < 45:
        ds = 0.8; dt = "🟡 Un peu court — idéal = 45-90s"
    elif 90 < dur <= 120:
        ds = 0.8; dt = "🟡 Un peu long — idéal = 45-90s"
    elif dur < 30:
        ds = 0.5; dt = "⚠️ Trop court pour TikTok"
    else:
        ds = 0.6; dt = "⚠️ Trop long — découpe en 2"
    details["duree"] = {"score": ds, "tip": dt}
    total += ds * 20

    # Accroche (8 premières secondes)
    hook_segs = [s for s in segments if start <= s["start"] <= start + 8]
    hook_text = " ".join(s["text"] for s in hook_segs).lower()
    hook_words = ["attention","regarde","jamais","secret","incroyable","vérité","choc",
                  "wait","stop","wow","shocking","never","hidden","watch","fou","impossible"]
    hook_hit = sum(1 for w in hook_words if w in hook_text)
    if hook_hit >= 2:
        hs = 1.0; ht = "✅ Accroche forte dès le début"
    elif hook_hit == 1:
        hs = 0.7; ht = "🟡 Accroche moyenne — commence par un mot fort"
    else:
        hs = 0.4; ht = "❌ Accroche faible — mets un texte choc en intro"
    details["accroche"] = {"score": hs, "tip": ht}
    total += hs * 25

    # Densité de parole
    clip_segs = [s for s in segments if start <= s["start"] <= end]
    words_min = len(" ".join(s["text"] for s in clip_segs).split()) / max(dur / 60, 0.1)
    if words_min >= 100:
        ss = 1.0; st = "✅ Rythme de parole parfait"
    elif words_min >= 60:
        ss = 0.75; st = "🟡 Rythme correct"
    elif words_min >= 25:
        ss = 0.5; st = "⚠️ Peu de paroles"
    else:
        ss = 0.3; st = "❌ Quasi pas de paroles — ajoute une voix off ?"
    details["parole"] = {"score": ss, "tip": st}
    total += ss * 20

    # Engagement (?, !)
    full_text = " ".join(s["text"] for s in clip_segs)
    eng = min(1.0, full_text.count("?") * 0.12 + full_text.count("!") * 0.08 + base * 0.5)
    et = "✅ Contenu engageant" if eng >= 0.7 else "🟡 Ajoute des questions pour les commentaires"
    details["engagement"] = {"score": eng, "tip": et}
    total += eng * 20

    # Énergie audio
    details["energie"] = {"score": base, "tip": "✅ Énergie forte" if base >= 0.7 else "🟡 Énergie correcte"}
    total += base * 15

    pct = round(total)
    if pct >= 85:   grade, gc = "A", "#00dfa2"
    elif pct >= 70: grade, gc = "B", "#5b8dee"
    elif pct >= 55: grade, gc = "C", "#f5c842"
    elif pct >= 40: grade, gc = "D", "#ff8c00"
    else:           grade, gc = "F", "#ff2d55"

    return {"grade": grade, "grade_color": gc, "tiktok_score": pct, "details": details}

# ─────────────────────────────────────────────────────────────
# 7. GÉNÉRATION SRT (sous-titres)
# ─────────────────────────────────────────────────────────────
def _generate_srt(segments, clip_start, clip_end, out_srt) -> bool:
    try:
        clip_segs = [s for s in segments
                     if s["start"] >= clip_start - 0.5 and s["end"] <= clip_end + 0.5]
        if not clip_segs:
            return False

        def ts(sec):
            sec = max(0, sec)
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

        lines = []
        for i, seg in enumerate(clip_segs, 1):
            s = max(0, seg["start"] - clip_start)
            e = min(clip_end - clip_start, seg["end"] - clip_start)
            text = seg["text"].strip()
            if not text:
                continue
            # Coupe en 2 lignes si trop long
            if len(text) > 42:
                mid = len(text) // 2
                sp  = text.rfind(" ", 0, mid)
                if sp > 0:
                    text = text[:sp] + "\n" + text[sp + 1:]
            lines.append(f"{i}\n{ts(s)} --> {ts(e)}\n{text}\n")

        if lines:
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return True
        return False
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────
# 8. EXPORT TIKTOK 1080×1920 + sous-titres + watermark
# ─────────────────────────────────────────────────────────────
def _video_has_audio(video_path: str) -> bool:
    """Vérifie si la vidéo a au moins une piste audio."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", str(video_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        info = json.loads(r.stdout)
        return any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    except Exception:
        return True  # en cas de doute, on suppose qu'il y a de l'audio


def _export_clip(input_path, start, end, out_path, srt_path=None, watermark=None, has_audio: bool = True) -> bool:
    """
    Export un clip TikTok 1080x1920.
    Le filter_complex est construit séquentiellement — pas de str.replace
    sur les labels, évite toute collision de label FFmpeg.
    """
    dur  = end - start
    fade = max(0, dur - 2.0)

    # Étape 1 : base — fond flouté + vidéo centrée + fade out
    filters = [
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,boxblur=20:20,eq=brightness=-0.4[bg]",
        "[0:v]scale=1080:-2[fg]",
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fade=t=out:st={fade}:d=2[v0]",
    ]
    current = "[v0]"
    idx = 1

    # Étape 2 : sous-titres (optionnel)
    if srt_path and Path(srt_path).exists():
        # Échappe le chemin SRT pour le parser FFmpeg filter
        safe_srt = str(srt_path).replace("\\", "/").replace("'", "\\'").replace(":", "\\:")
        nxt = f"[v{idx}]"
        filters.append(
            f"{current}subtitles='{safe_srt}':force_style='"
            f"FontName=Arial,FontSize=14,Bold=1,"
            f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
            f"Outline=2,Shadow=1,Alignment=2,MarginV=60'{nxt}"
        )
        current = nxt
        idx += 1

    # Étape 3 : watermark texte (optionnel)
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
        idx += 1

    vf = ";".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", str(input_path),
        "-t", str(dur),
        "-filter_complex", vf,
        "-map", current,
    ]
    if has_audio:
        cmd += ["-map", "0:a", "-af", f"afade=t=out:st={fade}:d=2",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    return r.returncode == 0

# ─────────────────────────────────────────────────────────────
# 9. CAPTION (sans Claude API)
# ─────────────────────────────────────────────────────────────
def _make_caption(clip) -> str:
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
# 10. PIPELINE PRINCIPAL
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

    async def upd(progress, message, **kw):
        set_job(job_id, progress=progress, message=message, **kw)
        await notify_ws(job_id)

    try:
        # ── Étape 1 : vérification ──────────────────────────────
        size_mb = int(Path(video_path).stat().st_size / 1024 / 1024)
        title   = Path(filename).stem[:60]
        set_job(job_id, video_title=title)
        await upd(5, f"✅ Vidéo reçue ({size_mb} MB) — vérification...")

        # Utilise la durée déjà calculée lors de l'upload si disponible
        duration = settings.get("_known_duration") or                    await loop.run_in_executor(executor, _get_duration, video_path)
        if duration > MAX_VIDEO_DURATION:
            raise Exception(
                f"Vidéo trop longue ({int(duration//60)}min). "
                f"Maximum 60 min. Découpe avec VLC (gratuit) puis réenvoie."
            )

        mins = int(duration // 60)
        secs = int(duration % 60)
        await upd(8, f"⏱️ Durée : {mins}min {secs}s. Vérification compatibilité...")

        # ── Étape 1b : re-encode si HEVC/VP9 (compatibilité 100%) ──
        video_path = await loop.run_in_executor(executor, _ensure_h264, video_path, job_id, duration)
        await upd(12, "🔧 Format vidéo compatible. Analyse audio...")

        # ── Étape 2 : audio energy ──────────────────────────────
        times, energies, emotions, duration = await loop.run_in_executor(
            executor, _extract_audio_energy, video_path, 3.0, duration)
        await upd(28, f"🎵 Audio analysé ({mins}min). Transcription Whisper...")

        # ── Étape 3 : Whisper ───────────────────────────────────
        segments = await loop.run_in_executor(
            executor, _transcribe_full, video_path, duration, job_id, lang)
        seg_count = len(segments)
        await upd(58, f"✍️ {seg_count} segments transcrits. Détection des moments forts...")

        # ── Étape 4 : détection automatique intelligente ────────
        clips = await loop.run_in_executor(
            executor, _auto_detect_clips, times, energies, emotions, segments, duration)

        if not clips:
            raise Exception("Aucun moment fort détecté. Essaie avec une vidéo avec plus de paroles.")

        n_clips = len(clips)
        await upd(65, f"✅ {n_clips} moment(s) fort(s) détecté(s) automatiquement. Export TikTok...",
                  status="exporting")

        # ── Étape 5 : export ────────────────────────────────────
        # has_audio calculé une seule fois pour toute la vidéo (pas par clip)
        has_audio = await loop.run_in_executor(executor, _video_has_audio, video_path)
        exported = []
        for idx, clip in enumerate(clips):
            clip_name = f"Clip_Elite_{idx+1}"
            out_path  = job_dir / f"{clip_name}.mp4"
            srt_path  = str(job_dir / f"{clip_name}.srt") if subtitles else None
            caption   = _make_caption(clip)
            tt_score  = _tiktok_score(clip, segments, duration)

            if subtitles and srt_path:
                await loop.run_in_executor(
                    executor, _generate_srt, segments, clip["start"], clip["end"], srt_path)

            wm = watermark if watermark else None
            success = await loop.run_in_executor(
                executor, _export_clip, video_path,
                clip["start"], clip["end"], str(out_path), srt_path, wm, has_audio)

            if success:
                exported.append({
                    **clip,
                    "filename":     f"{clip_name}.mp4",
                    "url":          f"/outputs/{job_id}/{clip_name}.mp4",
                    "preview_url":  f"/outputs/{job_id}/{clip_name}.mp4",
                    "rank":         idx + 1,
                    "caption":      caption,
                    "video_title":  title,
                    "tiktok_score": tt_score["tiktok_score"],
                    "grade":        tt_score["grade"],
                    "grade_color":  tt_score["grade_color"],
                    "tt_details":   tt_score["details"],
                    "has_subtitles": subtitles,
                })
            pct = 65 + int(((idx + 1) / n_clips) * 32)
            await upd(pct, f"🎬 Export {idx+1}/{n_clips} ({int(clip['duration'])}s)...")

        # ── ZIP ─────────────────────────────────────────────────
        zip_url = None
        if exported:
            zip_path = job_dir / "tous_les_clips.zip"
            def make_zip():
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                    for c in exported:
                        p = job_dir / c["filename"]
                        if p.exists():
                            zf.write(p, c["filename"])
            await loop.run_in_executor(executor, make_zip)
            zip_url = f"/outputs/{job_id}/tous_les_clips.zip"

        set_job(job_id, status="done", progress=100, clips=exported,
                zip_url=zip_url,
                message=f"🎉 {len(exported)} clips viraux prêts !")
        await notify_ws(job_id)

    except Exception as e:
        set_job(job_id, status="error", message=f"❌ {str(e)}")
        await notify_ws(job_id)
    finally:
        # Nettoyage vidéo uploadée dans tous les cas
        # On supprime tous les fichiers temporaires connus pour ce job
        # Supprime la vidéo source
        cleanup_paths = [
            video_path,
            str(UPLOAD_DIR / f"{job_id}.mp4"),
            str(UPLOAD_DIR / f"{job_id}_h264.mp4"),
        ]
        for p in cleanup_paths:
            try:
                if Path(p).exists():
                    os.remove(p)
            except Exception:
                pass
        # Supprime les fichiers WAV temporaires Whisper (au cas où crash mid-transcription)
        import glob
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
        # Envoie l'état courant immédiatement
        if job_id in jobs:
            await websocket.send_text(json.dumps(jobs[job_id]))
        # Keepalive
        while True:
            await asyncio.sleep(15)
            try:
                await websocket.send_text(json.dumps({"ping": True}))
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_connections.get(job_id, []):
            ws_connections[job_id].remove(websocket)

# ─────────────────────────────────────────────────────────────
# ROUTES API
# ─────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, request: Request):
    # Anti-abus par IP
    client_ip = request.client.host if request.client else "unknown"
    now_ts    = time.time()
    ip_jobs[client_ip] = [t for t in ip_jobs[client_ip] if now_ts - t < 3600]
    if len(ip_jobs[client_ip]) >= MAX_JOBS_PER_IP:
        raise HTTPException(429, "Trop de vidéos envoyées. Attends 1h avant de réessayer.")
    ip_jobs[client_ip].append(now_ts)

    job_id   = str(uuid.uuid4())[:8]
    out_path = str(UPLOAD_DIR / f"{job_id}.mp4")

    # Récupère le nom depuis Content-Disposition
    content_disp = request.headers.get("content-disposition", "")
    filename = "video.mp4"
    if "filename=" in content_disp:
        try:
            raw = content_disp.split("filename=")[1].strip().strip('"\'')
            filename = sanitize_filename(raw) + ".mp4"
        except Exception:
            pass

    # ── Streaming vers disque — ne charge JAMAIS en RAM ─────
    # Vérifie les magic bytes du premier chunk (détection format réel)
    VALID_MAGIC = {
        b"\x00\x00\x00",          # MP4/MOV ftyp (offset 4, on check début)
        b"\x1aE\xdf\xa3",         # MKV/WEBM EBML header
        b"RIFF",                     # AVI
        b"OggS",                     # OGG video
        b"\x47",                    # MPEG-TS (0x47 = sync byte)
        b"FLV",                      # Flash Video
        b"\x00\x00\x01\xba",     # MPEG-PS
        b"\x00\x00\x01\xb3",     # MPEG video
    }
    written = 0
    first_chunk = True
    with open(out_path, "wb") as f:
        async for chunk in request.stream():
            if not chunk:
                continue
            # Vérification magic bytes sur le PREMIER chunk seulement
            if first_chunk:
                first_chunk = False
                # MP4/MOV: magic "ftyp" se trouve aux octets 4-8
                is_video = (
                    len(chunk) >= 12 and chunk[4:8] in (b"ftyp", b"free", b"mdat", b"moov", b"wide")
                    or len(chunk) >= 4 and chunk[:4] in VALID_MAGIC
                    or len(chunk) >= 3 and chunk[:3] in VALID_MAGIC
                    or len(chunk) >= 1 and chunk[:1] in VALID_MAGIC
                )
                if not is_video:
                    # Dernier recours: vérifie le Content-Type header
                    ct = request.headers.get("content-type", "")
                    if not (ct.startswith("video/") or ct == "application/octet-stream" or ct == ""):
                        f.close()
                        try: os.remove(out_path)
                        except: pass
                        raise HTTPException(400, "Ce fichier n'est pas une vidéo reconnue.")
            f.write(chunk)
            written += len(chunk)
            # Sécurité : rejette si > 3GB
            if written > 3 * 1024 * 1024 * 1024:
                f.close()
                try: os.remove(out_path)
                except: pass
                raise HTTPException(413, "Fichier trop lourd (max 3 GB).")

    if written < 1000:
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise HTTPException(400, "Fichier vide ou invalide.")

    # Vérifie l'intégrité du fichier après upload complet
    # (détecte les uploads interrompus qui ont quand même écrit > 1000 bytes)
    try:
        verify_dur = _get_duration(out_path)
        if verify_dur <= 0:
            raise ValueError("Durée nulle")
    except Exception as ve:
        try: os.remove(out_path)
        except: pass
        raise HTTPException(400, f"Fichier vidéo non lisible après upload : {ve}")

    settings = {
        "lang":      request.headers.get("x-lang", "auto"),
        "subtitles": request.headers.get("x-subtitles", "true").lower() == "true",
        "watermark": request.headers.get("x-watermark", ""),
    }

    jobs[job_id] = {
        "status": "queued", "progress": 2,
        "message": "⬆️ Fichier reçu, initialisation...",
        "clips": [], "created_at": time.time(), "video_title": "",
    }
    # Passe verify_dur pour éviter un 2ème ffprobe dans le pipeline
    settings["_known_duration"] = verify_dur

    background_tasks.add_task(process_video_job, job_id, out_path, filename, settings)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job introuvable.")
    return jobs[job_id]

@app.get("/api/jobs")
async def list_jobs():
    result = []
    for jid, job in jobs.items():
        if job.get("status") == "done" and job.get("clips"):
            result.append({"job_id": jid, "clips": job["clips"],
                           "created_at": job.get("created_at", 0)})
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result[:20]


# ─────────────────────────────────────────────────────────────
# NETTOYAGE AUTO (clips > 3h supprimés)
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def start_cleanup():
    asyncio.create_task(_cleanup_loop())

async def _cleanup_loop():
    while True:
        await asyncio.sleep(1800)  # toutes les 30min
        now = time.time()
        for jid in list(jobs.keys()):
            if now - jobs[jid].get("created_at", now) > 10800:  # 3h
                job_dir = OUTPUT_DIR / jid
                if job_dir.exists():
                    import shutil
                    shutil.rmtree(job_dir, ignore_errors=True)
                jobs.pop(jid, None)

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
