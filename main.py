# ═══════════════════════════════════════════════════════════════════
#  ClipViral PRO — Version complète avec Claude AI + Whisper
#  ✅ Claude Haiku (~0.015€/vidéo)   ✅ Whisper tiny (gratuit)
#  ✅ YouTube direct (yt-dlp)        ✅ Découpage auto 15min
#  ✅ Mode auto (Claude choisit)     ✅ Render Free compatible
#  ✅ Mot de passe simple            ✅ 0€ hébergement
# ═══════════════════════════════════════════════════════════════════
import os, gc, glob, uuid, shutil, subprocess, json, time, asyncio
import random, zipfile, re, httpx
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ClipViral PRO API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_VIDEO_DURATION  = 3600
MAX_FILE_SIZE       = 2 * 1024 * 1024 * 1024
CLIP_MIN            = 61
CLIP_MAX            = 180
CLIP_GAP            = 90
WHISPER_SEGMENT_MAX = 900   # 15min max par segment
SITE_PASSWORD       = os.getenv("SITE_PASSWORD", "clipviral2025")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")

jobs: dict           = {}
ws_connections: dict = defaultdict(list)
ip_jobs: dict        = defaultdict(list)
MAX_JOBS_PER_IP      = 5
executor             = ThreadPoolExecutor(max_workers=1)
jobs: dict           = {}
jobs_created_at: dict = {}
ws_connections: dict = defaultdict(list)
ip_jobs: dict        = defaultdict(list)
MAX_JOBS_PER_IP      = 5
executor             = ThreadPoolExecutor(max_workers=1)
auto_executor        = ThreadPoolExecutor(max_workers=2)

import threading
_whisper_model = None
_whisper_lock  = threading.Lock()

def get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            try:
                import whisper as _w
                _whisper_model = _w.load_model("tiny")
                print("\u2705 Whisper tiny charg\u00e9 en m\u00e9moire")
            except Exception as e:
                print(f"Whisper import failed: {e}")
                _whisper_model = None
    return _whisper_model

_cost_lock  = threading.Lock()
_cost_data  = {"total_clips": 0, "total_cost_eur": 0.0, "total_analyses": 0}
COST_PER_CLIP = 0.013

def _add_cost(n_clips: int):
    with _cost_lock:
        _cost_data["total_clips"]    += n_clips
        _cost_data["total_analyses"] += 1
        _cost_data["total_cost_eur"] = round(_cost_data["total_clips"] * COST_PER_CLIP, 3)

CLIP_TTL_SECONDS = 2 * 3600  # nettoyage toutes les 2h

# ── Helpers ──────────────────────────────────────────────────────────
def set_job(job_id, **kw):
    if job_id in jobs: jobs[job_id].update(kw)

async def notify_ws(job_id):
    try: data = json.dumps(jobs.get(job_id, {}), default=str)
    except Exception:
        try:
            j = jobs.get(job_id, {})
            data = json.dumps({"status": j.get("status","unknown"), "progress": j.get("progress",0), "message":"..."})
        except: return
    dead = []
    for ws in list(ws_connections.get(job_id, [])):
        try: await ws.send_text(data)
        except: dead.append(ws)
    for d in dead:
        try: ws_connections[job_id].remove(d)
        except ValueError: pass

def sanitize_filename(name):
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'\s+', '_', name).strip('_.')
    return name[:80] or "video"

# ── Auth ─────────────────────────────────────────────────────────────
def _login_html():
    return """<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ClipViral PRO</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;600&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:#04040a;color:#f0f0ff;display:flex;align-items:center;justify-content:center;min-height:100vh;background:radial-gradient(ellipse 60% 60% at 50% 50%,rgba(124,58,237,.1),transparent 70%)}
.box{background:#0c0c1a;border:1px solid #1a1a30;border-radius:20px;padding:40px;width:360px;text-align:center}
.logo{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;margin-bottom:8px}
.logo em{font-style:normal;background:linear-gradient(135deg,#a855f7,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{font-size:13px;color:#7070a0;margin-bottom:28px}
input{width:100%;background:#111122;border:1px solid #1a1a30;border-radius:10px;padding:12px 16px;color:#f0f0ff;font-size:14px;outline:none;margin-bottom:14px;font-family:'DM Sans',sans-serif;transition:border-color .2s}
input:focus{border-color:rgba(124,58,237,.5)}
button{width:100%;background:linear-gradient(135deg,#7c3aed,#a855f7);color:#fff;border:none;border-radius:10px;padding:13px;font-size:14px;font-weight:700;cursor:pointer;box-shadow:0 4px 20px rgba(124,58,237,.35)}
.err{color:#f87171;font-size:12px;margin-top:10px;display:none}
</style></head><body>
<div class="box">
  <div class="logo">Clip<em>Viral</em> PRO</div>
  <div class="sub">Ton outil de clips TikTok IA</div>
  <input type="password" id="pw" placeholder="Mot de passe" onkeydown="if(event.key==='Enter')login()"/>
  <button onclick="login()">✦ Accéder</button>
  <div class="err" id="err">Mot de passe incorrect</div>
</div>
<script>
async function login(){
  const pw=document.getElementById('pw').value;
  const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
  if(r.ok){location.reload();}else{document.getElementById('err').style.display='block';}
}
</script></body></html>"""

@app.middleware("http")
async def auth_middleware(request, call_next):
    pub = ["/health","/favicon.ico","/api/login"]
    if any(request.url.path == p for p in pub):
        return await call_next(request)
    token = request.cookies.get("cv_token")
    if token != f"cv_{SITE_PASSWORD}_ok":
        if not request.url.path.startswith("/api"):
            return Response(content=_login_html(), media_type="text/html")
        raise HTTPException(status_code=401, detail="Non autorisé")
    return await call_next(request)

@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    if body.get("password") != SITE_PASSWORD:
        raise HTTPException(status_code=401, detail="Mot de passe incorrect")
    resp = Response(content=json.dumps({"ok": True}), media_type="application/json")
    resp.set_cookie("cv_token", f"cv_{SITE_PASSWORD}_ok", max_age=30*24*3600, httponly=True, samesite="lax")
    return resp

# ── Durée ─────────────────────────────────────────────────────────────
def _get_duration(video_path):
    try:
        r = subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_format",str(video_path)], capture_output=True, timeout=30)
        if r.returncode == 0:
            d = json.loads(r.stdout).get("format",{}).get("duration")
            if d: return float(d)
    except: pass
    try:
        r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(video_path)], capture_output=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip(): return float(r.stdout.strip())
    except: pass
    return 0.0

def _ensure_h264(video_path, job_id, duration):
    try:
        r = subprocess.run(["ffprobe","-v","quiet","-select_streams","v:0","-show_entries","stream=codec_name","-of","csv=p=0",str(video_path)], capture_output=True, timeout=20)
        codec = r.stdout.decode().strip().lower()
    except: codec = ""
    if codec in ("h264","avc","avc1",""): return video_path
    if duration > 1200: raise ValueError(f"Codec {codec} non supporté pour vidéos > 20min. Convertis en MP4 d'abord.")
    out = str(UPLOAD_DIR / f"{job_id}_h264.mp4")
    subprocess.run(["ffmpeg","-y","-i",str(video_path),"-c:v","libx264","-preset","ultrafast","-crf","28","-c:a","aac","-b:a","128k",out], capture_output=True, timeout=300)
    if Path(out).exists() and Path(out).stat().st_size > 0:
        try: os.remove(video_path)
        except: pass
        return out
    return video_path

def _has_audio(video_path):
    try:
        r = subprocess.run(["ffprobe","-v","quiet","-select_streams","a","-show_entries","stream=codec_name","-of","csv=p=0",str(video_path)], capture_output=True, timeout=15)
        return bool(r.stdout.strip())
    except: return True

# ── Découpage en segments ≤ 15min ────────────────────────────────────
def _split_into_segments(video_path, job_id, duration):
    if duration <= WHISPER_SEGMENT_MAX:
        return [(video_path, 0.0)]
    segments = []
    seg_dir = UPLOAD_DIR / f"{job_id}_segs"
    seg_dir.mkdir(exist_ok=True)
    n_segs = int(duration // WHISPER_SEGMENT_MAX) + (1 if duration % WHISPER_SEGMENT_MAX > 30 else 0)
    seg_dur = duration / n_segs
    for i in range(n_segs):
        start = i * seg_dur
        out   = str(seg_dir / f"seg_{i:03d}.mp4")
        subprocess.run(["ffmpeg","-y","-ss",str(start),"-i",str(video_path),"-t",str(seg_dur),"-c:v","copy","-c:a","copy",out], capture_output=True, timeout=120)
        if Path(out).exists() and Path(out).stat().st_size > 0:
            segments.append((out, start))
    return segments if segments else [(video_path, 0.0)]

# ── Whisper ───────────────────────────────────────────────────────────
def _transcribe_segment(video_path, offset=0.0):
    """Transcrit un segment audio — utilise le modèle Whisper persistant."""
    model = get_whisper_model()
    if model is None:
        return []
    try:
        audio_path = str(video_path) + "_audio.wav"
        subprocess.run(["ffmpeg","-y","-i",str(video_path),"-vn","-ac","1","-ar","16000","-f","wav",audio_path],
                       capture_output=True, timeout=120)
        if not Path(audio_path).exists(): return []
        result = model.transcribe(audio_path, language="fr", fp16=False, verbose=False)
        segs = [{"start":round(s["start"]+offset,1),"end":round(s["end"]+offset,1),"text":s["text"].strip()}
                for s in result.get("segments",[])]
        try: os.remove(audio_path)
        except: pass
        return segs
    except Exception as e:
        print(f"Whisper error: {e}")
        return []

# ── Analyse audio RMS ─────────────────────────────────────────────────
def _extract_audio_energy(video_path, seg_dur=3.0, duration=None):
    if not duration: duration = _get_duration(video_path)
    CHUNK = 300
    all_times, all_energies, all_emotions = [], [], []
    def _neutral(start, count):
        for i in range(count):
            all_times.append(start + i * seg_dur); all_energies.append(0.3); all_emotions.append(0.3)
    for chunk_start in range(0, int(duration), CHUNK):
        chunk_end = min(chunk_start + CHUNK, duration); chunk_dur = chunk_end - chunk_start
        n = max(1, int(chunk_dur / seg_dur))
        try:
            r = subprocess.run(["ffmpeg","-y","-ss",str(chunk_start),"-i",str(video_path),"-t",str(chunk_dur),"-vn","-ac","1","-ar","22050","-f","f32le","-"], capture_output=True, timeout=90)
        except subprocess.TimeoutExpired:
            _neutral(chunk_start, n); continue
        if not r.stdout or len(r.stdout) < 400:
            _neutral(chunk_start, n); continue
        try: samples = np.frombuffer(r.stdout, dtype=np.float32)
        except: _neutral(chunk_start, n); continue
        if len(samples) < 100: _neutral(chunk_start, n); continue
        sr = 22050; hop = int(seg_dur * sr)
        for i in range(0, len(samples) - hop, hop):
            seg = samples[i:i+hop]
            rms = float(np.sqrt(np.mean(seg**2))); variance = float(np.var(seg))
            half = len(seg)//2; delta = abs(float(np.sqrt(np.mean(seg[half:]**2))) - float(np.sqrt(np.mean(seg[:half]**2))))
            emotion = min(1.0, variance * 40 + delta * 3)
            all_times.append(chunk_start + i/sr); all_energies.append(rms); all_emotions.append(emotion)
        del samples
    if not all_times: all_times, all_energies, all_emotions = [0.0],[0.3],[0.3]
    return np.array(all_times), np.array(all_energies), np.array(all_emotions), float(duration)

# ── Détection clips ───────────────────────────────────────────────────
def _auto_detect_clips(times, energies, emotions, duration):
    norm_e = energies/(energies.max()+1e-9); norm_em = emotions/(emotions.max()+1e-9)
    combined = norm_e*0.65 + norm_em*0.35
    window = max(3, int(10/3))
    smoothed = np.convolve(combined, np.ones(window)/window, mode="same")
    mean_s=smoothed.mean(); std_s=smoothed.std(); cv=std_s/(mean_s+1e-6)
    k = 0.8 if cv>0.5 else (1.0 if cv>0.3 else 0.5)
    threshold = mean_s + k * std_s
    if   duration<=600:  max_clips=2
    elif duration<=900:  max_clips=3
    elif duration<=1800: max_clips=5
    elif duration<=3600: max_clips=8
    else:                max_clips=10
    peaks = [i for i in range(1, len(smoothed)-1) if smoothed[i]>=threshold and smoothed[i]>=smoothed[i-1] and smoothed[i]>=smoothed[i+1]]
    peaks.sort(key=lambda i: smoothed[i], reverse=True)
    ext_thresh = mean_s + 0.2*std_s; clips = []
    for pi in peaks:
        if len(clips) >= max_clips: break
        peak_t = float(times[pi])
        if any(abs(peak_t-(c["start"]+c["end"])/2) < CLIP_GAP for c in clips): continue
        left=pi
        while left>0 and (peak_t-times[left-1])<=CLIP_MAX and smoothed[left-1]>=ext_thresh: left-=1
        right=pi
        while right<len(times)-1:
            if (times[right+1]-times[left])>CLIP_MAX: break
            if smoothed[right+1]>=ext_thresh: right+=1
            elif (times[right]-times[left])<CLIP_MIN: right+=1
            else: break
        s=float(times[left]); e=float(times[right])
        if (e-s)<CLIP_MIN: e=min(s+CLIP_MIN, duration)
        if (e-s)>CLIP_MAX: s=max(0.0, peak_t-CLIP_MAX/2); e=min(duration, s+CLIP_MAX)
        if e>duration: e=duration; s=max(0.0, e-CLIP_MIN)
        if e-s<=0: continue
        mask=(times>=s)&(times<=e)
        vs=min(1.0, float(smoothed[mask].mean())/(smoothed.max()+1e-6)) if mask.any() else 0.5
        clips.append({"start":round(s,1),"end":round(e,1),"duration":round(e-s,1),"viral_score":round(vs,3),"reason":""})
    if not clips:
        n=min(3,max(1,int(duration/600))); step=duration/(n+1)
        for i in range(n):
            s=max(0.0,round(step*(i+1)-CLIP_MIN/2,1)); e=min(float(duration),s+CLIP_MIN)
            if e-s>0: clips.append({"start":round(s,1),"end":round(e,1),"duration":round(e-s,1),"viral_score":0.5,"reason":"Fallback"})
    clips.sort(key=lambda x: x["start"])
    return clips

# ── Score TikTok ──────────────────────────────────────────────────────
def _tiktok_score(clip, duration):
    dur=clip["end"]-clip["start"]; base=clip.get("viral_score",0.5); total=0.0; det={}
    if 45<=dur<=90: ds,dt=1.0,"✅ Durée parfaite TikTok (45-90s)"
    elif 30<=dur<45: ds,dt=0.8,"🟡 Un peu court — idéal 45-90s"
    elif 90<dur<=120: ds,dt=0.8,"🟡 Un peu long — idéal 45-90s"
    elif dur<30: ds,dt=0.5,"⚠️ Trop court"
    else: ds,dt=0.6,"⚠️ Trop long"
    det["duree"]={"score":ds,"tip":dt}; total+=ds*30
    pos_ratio=clip["start"]/max(duration,1)
    if pos_ratio<0.25: ps,pt=1.0,"✅ Début de vidéo — fort potentiel"
    elif pos_ratio<0.6: ps,pt=0.8,"🟡 Milieu de vidéo"
    else: ps,pt=0.6,"⚠️ Fin de vidéo"
    det["position"]={"score":ps,"tip":pt}; total+=ps*20
    det["energie"]={"score":base,"tip":"✅ Énergie forte" if base>=0.7 else "🟡 Énergie correcte" if base>=0.4 else "⚠️ Énergie faible"}
    total+=base*30; det["conseil"]={"score":0.7,"tip":"💡 Texte d'accroche dans les 3 premières secondes"}; total+=0.7*20
    pct=round(total)
    if pct>=85: grade,gc="A","#00dfa2"
    elif pct>=70: grade,gc="B","#a855f7"
    elif pct>=55: grade,gc="C","#f5c842"
    elif pct>=40: grade,gc="D","#ff8c00"
    else: grade,gc="F","#ff2d55"
    return {"grade":grade,"grade_color":gc,"tiktok_score":pct,"details":det}

# ── Export TikTok 1080×1920 ───────────────────────────────────────────
def _export_clip(input_path, start, end, out_path, watermark=None, has_audio=True):
    dur=end-start
    if dur<=0: return False
    fade=max(0.0, dur-2.0)
    vf_parts=["scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos","pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",f"fade=t=in:st=0:d=0.4,fade=t=out:st={fade:.2f}:d=0.4"]
    if watermark:
        safe=watermark.replace("'","").replace("\\","")[:30]
        vf_parts.append(f"drawtext=text='{safe}':fontsize=28:fontcolor=white@0.75:x=20:y=20:shadowcolor=black@0.6:shadowx=1:shadowy=1:font=DejaVuSans-Bold")
    cmd=["ffmpeg","-y","-ss",str(start),"-i",str(input_path),"-t",str(dur),"-vf",",".join(vf_parts),"-c:v","libx264","-preset","ultrafast","-crf","26","-threads","1","-pix_fmt","yuv420p","-movflags","+faststart"]
    if has_audio: cmd+=["-c:a","aac","-b:a","128k","-ac","2"]
    else: cmd+=["-an"]
    cmd.append(str(out_path))
    try:
        r=subprocess.run(cmd, capture_output=True, timeout=240)
        return r.returncode==0 and Path(out_path).exists() and Path(out_path).stat().st_size>0
    except subprocess.TimeoutExpired: return False

# ── Claude Haiku ─────────────────────────────────────────────────────
async def _claude_select_best_clips(all_clips, transcript_segments, video_title, niche, duration):
    """
    Étape 1 — Claude Haiku lit TOUTE la transcription et choisit
    les meilleurs moments parmi les candidats audio détectés.
    Retourne la liste triée avec un score IA par clip.
    """
    if not ANTHROPIC_API_KEY or not transcript_segments:
        return all_clips  # sans transcription → on garde l'ordre audio

    # Résumé de chaque candidat avec sa transcription
    clips_summary = []
    for i, clip in enumerate(all_clips):
        relevant = [s for s in transcript_segments
                    if s["end"] >= clip["start"] and s["start"] <= clip["end"]]
        text = " ".join(s["text"] for s in relevant[:25]).strip()
        clips_summary.append({
            "id": i,
            "debut": f"{int(clip['start']//60)}:{int(clip['start']%60):02d}",
            "fin":   f"{int(clip['end']//60)}:{int(clip['end']%60):02d}",
            "duree": int(clip["duration"]),
            "energie_audio": int(clip.get("viral_score", 0.5) * 100),
            "transcription": text or "Non disponible"
        })

    VIRAL_PATTERNS = """
MOMENTS QUI CARTONNENT SUR TIKTOK :
- Affirmation choquante ou contre-intuitive ("La motivation c'est du bullshit")
- Chiffre frappant ou révélation ("J'ai fait 0€ puis 40k en un mois")
- Début d'histoire avec tension ("Le jour où j'ai tout perdu...")
- Question rhétorique forte ("Tu vas vraiment passer ta vie à attendre ?")
- Conseil court et actionnable immédiatement applicable
- Émotion authentique : rire nerveux, silence pesant, voix qui tremble

MOMENTS À ÉVITER :
- Transitions entre sujets ("Donc comme je disais...")
- Contexte sans conclusion ("C'est une longue histoire...")
- Bavardage social sans contenu
- Fin de phrase qui dépend du contexte précédent
- Introductions ou présentations"""

    system = f"""Tu es un expert en contenu viral TikTok francophone, niche {niche}.
Tu sais exactement quels moments d'une vidéo vont exploser sur TikTok.
Réponds UNIQUEMENT en JSON valide, sans markdown, sans explication.
{VIRAL_PATTERNS}"""

    user = f"""Vidéo : "{video_title}" ({int(duration//60)}min, niche {niche})

Voici {len(clips_summary)} candidats détectés par analyse audio.
Classe-les du MEILLEUR au moins bon pour TikTok en te basant sur la transcription.
Donne un score_ia entre 0 et 100 pour chaque clip.

CANDIDATS :
{json.dumps(clips_summary, ensure_ascii=False, indent=2)}

JSON requis (tous les clips, triés du meilleur au pire) :
{{"classement": [{{"id": 0, "score_ia": 85, "raison_courte": "Affirmation choquante sur la motivation"}}]}}"""

    try:
        async with httpx.AsyncClient(timeout=35) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 400,
                      "system": system, "messages": [{"role": "user", "content": user}]}
            )
        if resp.status_code != 200: return all_clips
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r'^```json\s*', '', raw); raw = re.sub(r'\s*```$', '', raw)
        data = json.loads(raw)
        classement = data.get("classement", [])

        # Réordonne les clips selon le jugement IA
        id_to_score = {item["id"]: (item.get("score_ia", 50), item.get("raison_courte","")) for item in classement}
        for i, clip in enumerate(all_clips):
            score_ia, raison = id_to_score.get(i, (int(clip.get("viral_score",0.5)*100), ""))
            clip["score_ia"]    = score_ia
            clip["raison_ia"]   = raison

        sorted_clips = sorted(all_clips, key=lambda c: c.get("score_ia", 0), reverse=True)
        return sorted_clips

    except Exception as e:
        print(f"Claude selection error: {e}")
        return all_clips


async def _claude_analyze(clip, video_title, transcript_segments, niche="mindset", rank=1):
    """
    Étape 2 — Claude Haiku génère le contenu TikTok complet pour un clip déjà sélectionné.
    Titre punchy · Hook d'accroche · Caption · Hashtags · Timing
    """
    if not ANTHROPIC_API_KEY:
        return _fallback_analysis(clip, video_title, niche, rank)

    # Transcription du clip uniquement
    clip_transcript = ""
    if transcript_segments:
        relevant = [s for s in transcript_segments if s["end"] >= clip["start"] and s["start"] <= clip["end"]]
        clip_transcript = " ".join(s["text"] for s in relevant[:30]).strip()

    dur = int(clip["duration"])
    score_ia = clip.get("score_ia", int(clip.get("viral_score", 0.5) * 100))
    raison_ia = clip.get("raison_ia", "")
    total_dur = clip.get("_total_dur", 600)
    pos = "début" if clip["start"]/max(total_dur,1) < 0.33 else "milieu" if clip["start"]/max(total_dur,1) < 0.66 else "fin"

    # Hashtags par niche
    HASHTAGS_NICHE = {
        "mindset":  ["#mindset","#motivation","#developpementpersonnel","#fyp","#viral","#success","#mentality","#pourtoi"],
        "business": ["#business","#entrepreneur","#argent","#finance","#fyp","#viral","#richesse","#pourtoi"],
        "podcast":  ["#podcast","#interview","#culture","#france","#fyp","#viral","#debat","#pourtoi"],
        "gaming":   ["#gaming","#gamer","#jeux","#fyp","#viral","#france","#twitch","#pourtoi"],
        "humour":   ["#humour","#drole","#rire","#fyp","#viral","#france","#comedie","#pourtoi"],
        "food":     ["#food","#cuisine","#recette","#fyp","#viral","#france","#foodtok","#pourtoi"],
        "sport":    ["#sport","#fitness","#training","#fyp","#viral","#france","#musculation","#pourtoi"],
        "trending": ["#trending","#tendance","#fyp","#viral","#france","#pourtoi","#explore","#foryou"],
    }
    base_hashtags = HASHTAGS_NICHE.get(niche, HASHTAGS_NICHE["trending"])

    # Timing optimal par niche
    TIMING_NICHE = {
        "mindset":  "Jeudi ou vendredi · 19h-21h (audience 18-35 ans après le travail)",
        "business": "Lundi ou mardi · 7h-9h (entrepreneurs le matin) ou 20h-22h",
        "podcast":  "Dimanche · 15h-18h (écoute détendue le week-end)",
        "gaming":   "Vendredi soir ou samedi · 20h-23h (peak gaming)",
        "humour":   "Mercredi ou vendredi · 18h-22h (détente en fin de semaine)",
        "food":     "Samedi ou dimanche matin · 10h-12h (inspiration repas)",
        "sport":    "Lundi · 6h-8h (motivation début de semaine) ou 18h-20h",
        "trending": "Mercredi ou jeudi · 18h-21h (pic d'activité TikTok FR)",
    }
    timing_conseil = TIMING_NICHE.get(niche, TIMING_NICHE["trending"])

    system = f"""Tu es un expert en contenu viral TikTok francophone, spécialisé niche {niche}.
Tu crées des titres et hooks qui font ARRÊTER le scroll en moins de 2 secondes.
Réponds UNIQUEMENT en JSON valide, sans markdown."""

    user = f"""Génère le contenu TikTok pour ce clip déjà sélectionné comme excellent.

VIDÉO : "{video_title}"
CLIP #{rank} : {clip['start']}s → {clip['end']}s ({dur}s) · position {pos}
SCORE IA : {score_ia}/100{f' — {raison_ia}' if raison_ia else ''}
TRANSCRIPTION : "{clip_transcript or 'Non disponible'}"

RÈGLES TITRE : accrocheur, max 55 chars, commence par un verbe fort ou chiffre ou emoji
RÈGLES HOOK : ce que tu lis/entends dans les 2 premières secondes, doit créer une question dans la tête du viewer
RÈGLES CAPTION : 3-4 lignes max, emojis naturels, appel à l'action simple

JSON requis :
{{"titre": "Titre qui stoppe le scroll (max 55 chars)", "hook": "Phrase des 2 premières secondes (max 70 chars)", "caption": "Caption complète prête à coller\\n\\nLigne 2\\n\\nLigne 3", "score_viral": {score_ia}}}"""

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 500,
                      "system": system, "messages": [{"role": "user", "content": user}]}
            )
        if resp.status_code != 200: return _fallback_analysis(clip, video_title, niche, rank)
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r'^```json\s*', '', raw); raw = re.sub(r'\s*```$', '', raw)
        data = json.loads(raw)

        # Caption enrichie avec hashtags
        caption_base = data.get("caption", "")
        caption_full = f"{caption_base}\n\n{' '.join(base_hashtags)}" if caption_base else ""

        return {
            "titre":       data.get("titre", f"Moment #{rank} — {video_title[:25]}"),
            "hook":        data.get("hook", ""),
            "caption":     caption_full,
            "hashtags":    base_hashtags,
            "score_viral": int(data.get("score_viral", score_ia)),
            "raison":      raison_ia or f"Score IA {score_ia}/100",
            "timing":      timing_conseil,
            "ai_powered":  True,
        }

    except Exception as e:
        print(f"Claude analyze error: {e}")
        return _fallback_analysis(clip, video_title, niche, rank)

def _fallback_analysis(clip, video_title, niche="mindset", rank=1):
    vs=int(clip.get("viral_score",0.5)*100); dur=int(clip["duration"])
    s=int(clip["start"]//60); sm=int(clip["start"]%60); e=int(clip["end"]//60); em=int(clip["end"]%60)
    NICHES={"mindset":{"titre":f"Ce moment va changer ta façon de penser 🧠","hook":"Arrête de scroller — écoute ça","hashtags":["#mindset","#motivation","#developpementpersonnel","#fyp","#viral","#success","#mentality","#pourtoi","#citation","#croissance"]},"business":{"titre":f"La vérité que personne te dit sur l'argent 💰","hook":"Ce passage a changé ma vie","hashtags":["#business","#entrepreneur","#argent","#finance","#fyp","#viral","#richesse","#pourtoi","#investissement","#success"]},"default":{"titre":f"Moment #{rank} — {video_title[:30]}","hook":"Tu dois voir ça 👀","hashtags":["#fyp","#viral","#pourtoi","#trending","#france","#interessant","#worth","#watch","#content","#tiktok"]}}
    n=NICHES.get(niche,NICHES["default"])
    return {"titre":n["titre"],"hook":n["hook"],"caption":f"{n['hook']}\n\n📍 {s}:{sm:02d}→{e}:{em:02d} · {dur}s\n\n{' '.join(n['hashtags'][:6])}","hashtags":n["hashtags"],"score_viral":vs,"raison":f"Pic d'énergie audio {vs}% — {dur}s optimal","timing":"Poste 18h-21h jeudi-vendredi","ai_powered":False}

def _make_title(clip, rank, video_title="", ai_titre=""):
    if ai_titre: return ai_titre
    s=int(clip["start"]//60); sm=int(clip["start"]%60); e=int(clip["end"]//60); em=int(clip["end"]%60)
    vs=int(clip.get("viral_score",0.5)*100)
    label="Moment Viral" if vs>=80 else "Moment Fort" if vs>=65 else "Moment Clé" if vs>=50 else "Extrait"
    timing=f"{s}:{sm:02d}→{e}:{em:02d}"
    if video_title:
        short=video_title[:30]+("…" if len(video_title)>30 else "")
        return f"{label} #{rank} | {short} ({timing})"
    return f"{label} #{rank} ({timing}) · {int(clip['duration'])}s"

def _make_hashtags_fallback(video_title, niche=""):
    base=["#fyp","#pourtoi","#viral","#tiktok","#foryou"]
    t=(video_title+" "+niche).lower()
    if any(k in t for k in ["mindset","motivation","discipline","success","mental","reussite"]): extra=["#mindset","#motivation","#developpementpersonnel","#success","#mentality"]
    elif any(k in t for k in ["business","entrepreneur","argent","finance","invest"]): extra=["#business","#entrepreneur","#finance","#argent","#investissement"]
    elif any(k in t for k in ["podcast","interview","debat"]): extra=["#podcast","#interview","#france","#culture","#interessant"]
    else: extra=["#interesting","#watch","#explore","#content","#trending"]
    return base+extra[:5]

# ── YouTube download ──────────────────────────────────────────────────
async def _download_youtube(url, job_id):
    out_tmpl=str(UPLOAD_DIR/f"{job_id}_yt.%(ext)s")
    try:
        r=subprocess.run(["yt-dlp","--no-download","--print","%(title)s|||%(duration)s","--no-playlist",url], capture_output=True, text=True, timeout=30)
        if r.returncode==0 and "|||" in r.stdout:
            parts=r.stdout.strip().split("|||")
            yt_title=parts[0][:80] if parts else "YouTube Vidéo"
            yt_dur=float(parts[1]) if len(parts)>1 else 0
        else: yt_title="YouTube Vidéo"; yt_dur=0
    except: yt_title="YouTube Vidéo"; yt_dur=0
    if yt_dur>MAX_VIDEO_DURATION: raise ValueError(f"Vidéo trop longue ({int(yt_dur//60)}min). Maximum 60min.")
    proc=subprocess.run(["yt-dlp","-f","bestvideo[height<=480]+bestaudio/best[height<=480]/best","--merge-output-format","mp4","--no-playlist","--output",out_tmpl,url], capture_output=True, timeout=300)
    if proc.returncode!=0: raise ValueError(f"Téléchargement impossible: {proc.stderr.decode()[-200:]}")
    candidates=list(UPLOAD_DIR.glob(f"{job_id}_yt.*"))
    if not candidates: raise ValueError("Fichier téléchargé introuvable")
    video_path=str(candidates[0])
    if not yt_dur: yt_dur=_get_duration(video_path)
    return video_path, yt_title, yt_dur

# ── Mode Auto — Étape 1 : Claude génère les tendances ────────────────
async def _auto_get_trends(niche):
    """
    Claude Haiku génère 6 tendances thématiques chaudes pour la niche.
    Pas d'URLs — juste des thèmes. yt-dlp cherche ensuite les vraies vidéos.
    Coût : ~0.001€ par appel.
    """
    FALLBACK_TRENDS = {
        "mindset":  [
            {"theme": "Discipline vs Motivation", "emoji": "🧠", "why": "Le débat qui cartonne en ce moment"},
            {"theme": "Quiet quitting expliqué",  "emoji": "😮", "why": "Tendance pro virale cette semaine"},
            {"theme": "Dopamine detox en 2025",   "emoji": "📵", "why": "Sujet evergreen qui revient fort"},
            {"theme": "Stoïcisme pour débutants", "emoji": "⚡", "why": "Niche under-exploitée en FR"},
            {"theme": "Habitudes des millionnaires","emoji":"💰","why": "Curiosité universelle"},
            {"theme": "Sortir de sa zone de confort","emoji":"🚀","why":"Accroche émotionnelle forte"},
        ],
        "business": [
            {"theme": "Side hustle en 2025",      "emoji": "💻", "why": "Recherche top cette semaine"},
            {"theme": "Dropshipping mort ou vivant","emoji":"📦","why": "Controverse = engagement"},
            {"theme": "IA pour entrepreneurs",    "emoji": "🤖", "why": "Sujet ultra-trending"},
            {"theme": "Premier 1000€ en ligne",   "emoji": "💶", "why": "Dream outcome universel"},
            {"theme": "Erreurs des entrepreneurs", "emoji":"❌", "why": "Échec = contenu très partagé"},
            {"theme": "LinkedIn vs TikTok pro",   "emoji": "📱", "why": "Débat plateforme viral"},
        ],
        "podcast":  [
            {"theme": "Interview mindset français","emoji": "🎙️","why": "Format long = clips riches"},
            {"theme": "Débat société 2025",       "emoji": "🔥", "why": "Controverse = views"},
            {"theme": "Success stories françaises","emoji": "⭐","why": "Inspiration locale"},
            {"theme": "Santé mentale jeunes adultes","emoji":"💚","why":"Sujet universel très partagé"},
            {"theme": "Tech et société",           "emoji": "💡","why": "Croissance rapide"},
            {"theme": "Révélations business FR",  "emoji": "😱", "why": "Clickbait naturel"},
        ],
    }

    if not ANTHROPIC_API_KEY:
        return FALLBACK_TRENDS.get(niche, FALLBACK_TRENDS["mindset"])

    system = "Tu es un expert en tendances TikTok francophone. Réponds UNIQUEMENT en JSON valide, sans markdown."
    user = f"""Génère 6 thèmes tendance actuellement sur TikTok FR pour la niche "{niche}".
Ces thèmes doivent être : chauds cette semaine, clipables (moments forts possibles), francophones.

JSON requis :
{{"tendances": [{{"theme": "Nom du thème court", "emoji": "🔥", "why": "Pourquoi ça cartonne en 1 phrase courte"}}]}}"""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 400,
                      "system": system, "messages": [{"role": "user", "content": user}]}
            )
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r'^```json\s*', '', raw); raw = re.sub(r'\s*```$', '', raw)
        return json.loads(raw).get("tendances", [])[:6]
    except Exception as e:
        print(f"Trends error: {e}")
        return FALLBACK_TRENDS.get(niche, FALLBACK_TRENDS["mindset"])


# ── Mode Auto — Étape 2 : yt-dlp cherche les vraies vidéos ───────────
def _search_youtube_for_theme(theme, niche, max_results=4):
    """
    yt-dlp cherche les vidéos YouTube correspondant au thème.
    Filtre : francophones, > 10min (pour avoir des clips), récentes.
    Retourne liste de {url, titre, duree, chaine, vues}.
    """
    query = f"{theme} {niche} français"
    try:
        r = subprocess.run([
            "yt-dlp",
            f"ytsearch{max_results}:{query}",
            "--no-download",
            "--print", "%(webpage_url)s|||%(title)s|||%(duration)s|||%(uploader)s|||%(view_count)s",
            "--match-filter", "duration > 600",   # > 10min
            "--no-playlist",
        ], capture_output=True, text=True, timeout=25)

        videos = []
        for line in r.stdout.strip().splitlines():
            if "|||" not in line: continue
            parts = line.split("|||")
            if len(parts) < 4: continue
            try:
                dur = int(float(parts[2])) if parts[2] else 0
                views = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
                if dur < 600: continue  # double filtre
                videos.append({
                    "url":    parts[0].strip(),
                    "titre":  parts[1].strip()[:80],
                    "duree":  f"{dur//60}min",
                    "chaine": parts[3].strip()[:40],
                    "vues":   f"{views//1000}k" if views >= 1000 else str(views),
                })
            except: continue
        return videos[:4]
    except Exception as e:
        print(f"yt-dlp search error: {e}")
        return []

# ── Job principal ─────────────────────────────────────────────────────
async def process_video_job(job_id, video_path, filename, settings=None):
    if settings is None: settings={}
    job_dir=OUTPUT_DIR/job_id; job_dir.mkdir(exist_ok=True)
    loop=asyncio.get_running_loop()
    watermark=settings.get("watermark","").strip()[:40]
    niche=settings.get("niche","mindset")
    use_whisper=settings.get("whisper",True)

    async def upd(pct, msg, **kw):
        set_job(job_id, progress=pct, message=msg, **kw); await notify_ws(job_id)

    try:
        size_mb=Path(video_path).stat().st_size//(1024*1024)
        title=sanitize_filename(Path(filename).stem)
        set_job(job_id, video_title=title)
        await upd(5, f"✅ Vidéo reçue ({size_mb}MB)...")
        duration=float(settings.get("_known_duration") or await loop.run_in_executor(executor,_get_duration,video_path))
        if duration>MAX_VIDEO_DURATION:
            m,s=int(duration//60),int(duration%60)
            raise ValueError(f"Vidéo trop longue : {m}min {s}s. Maximum 60min.")
        mins=int(duration//60)
        await upd(8, f"⏱️ Durée : {mins}min. Format...")
        video_path=await loop.run_in_executor(executor,_ensure_h264,video_path,job_id,duration)
        await upd(12, "🔧 Format OK. Découpage segments...")

        # Découpage en segments ≤ 15min
        segments=await loop.run_in_executor(executor,_split_into_segments,video_path,job_id,duration)
        n_segs=len(segments)
        await upd(15, f"✂️ {n_segs} segment(s). {'Transcription Whisper...' if use_whisper else 'Analyse audio...'}")

        # Whisper
        all_transcript=[]
        if use_whisper:
            for si,(seg_path,offset) in enumerate(segments):
                pct_w=15+int((si/n_segs)*22)
                await upd(pct_w, f"🎤 Whisper segment {si+1}/{n_segs}...")
                seg_transcript=await loop.run_in_executor(executor,_transcribe_segment,seg_path,offset)
                all_transcript.extend(seg_transcript)
            await upd(37, f"✅ Transcription OK ({len(all_transcript)} segments). Analyse audio...")
        else:
            await upd(37, "🎵 Analyse audio...")

        # Analyse audio RMS
        audio_done=asyncio.Event()
        async def _audio_hb():
            icons=["🎵","🎶","🎤","🔊"]; i=0; pct=37
            while not audio_done.is_set():
                await asyncio.sleep(20)
                if audio_done.is_set(): break
                pct=min(55,pct+3); await upd(pct, f"{icons[i%4]} Analyse audio... {pct}%"); i+=1
        hb=asyncio.create_task(_audio_hb())
        try:
            times,energies,emotions,duration=await loop.run_in_executor(executor,_extract_audio_energy,video_path,3.0,duration)
        finally:
            audio_done.set(); hb.cancel()
            try: await hb
            except asyncio.CancelledError: pass

        await upd(56, "✅ Audio analysé. Détection moments forts...")
        clips=await loop.run_in_executor(executor,_auto_detect_clips,times,energies,emotions,duration)
        if not clips: raise ValueError("Aucun moment fort détecté.")
        for c in clips: c["_total_dur"]=duration

        # ── Claude juge les candidats AVANT de générer le contenu ─────────
        await upd(59, f"🤖 Claude évalue {len(clips)} candidats audio...")
        clips = await _claude_select_best_clips(clips, all_transcript, title, niche, duration)
        clips = clips[:5]  # max 5 clips, les meilleurs selon Claude
        await upd(63, f"✅ {len(clips)} moment(s) retenus. Génération contenu IA...")

        has_audio=await loop.run_in_executor(executor,_has_audio,video_path)
        exported=[]

        for i,clip in enumerate(clips):
            # ✅ Nom de fichier basé sur le titre IA si disponible
            ai_pre = clip.get("ai_titre","") or clip.get("titre","")
            if ai_pre:
                safe_name = re.sub(r"[^\w\s\-]","", ai_pre)[:35].strip().replace(" ","_")
                name = f"{safe_name}_{i+1}" if safe_name else f"Clip_Elite_{i+1}"
            else:
                name = f"Clip_Elite_{i+1}"
            out_path=str(job_dir/f"{name}.mp4"); dur_s=int(clip["duration"])
            pct_base=63+int(i/len(clips)*28)
            await upd(pct_base, f"🤖 Claude génère contenu clip {i+1}/{len(clips)}...")
            ai=await _claude_analyze(clip,title,all_transcript,niche,i+1)
            await upd(pct_base+1, f"🎬 Export clip {i+1}/{len(clips)} ({dur_s}s)...")

            export_done=asyncio.Event()
            async def _exp_hb(pb=pct_base+1,n=i+1,tot=len(clips),ds=dur_s):
                secs=0
                while not export_done.is_set():
                    await asyncio.sleep(12)
                    if export_done.is_set(): break
                    secs+=12; await upd(pb, f"🎬 Export clip {n}/{tot} ({ds}s) — {secs}s...")
            hb_exp=asyncio.create_task(_exp_hb())
            try:
                ok=await loop.run_in_executor(executor,_export_clip,video_path,clip["start"],clip["end"],out_path,watermark or None,has_audio)
            finally:
                export_done.set(); hb_exp.cancel()
                try: await hb_exp
                except asyncio.CancelledError: pass

            if ok:
                score=_tiktok_score(clip,duration)
                hashtags=ai.get("hashtags") or _make_hashtags_fallback(title,niche)
                clip_title=_make_title(clip,i+1,title,ai.get("titre",""))
                caption=ai.get("caption") or f"{ai.get('hook','')}\n\n{' '.join(hashtags[:6])}"
                exported.append({**clip,"filename":f"{name}.mp4","url":f"/outputs/{job_id}/{name}.mp4","rank":i+1,"clip_title":clip_title,"hook":ai.get("hook",""),"caption":caption,"hashtags":hashtags,"raison":ai.get("raison",""),"timing":ai.get("timing",""),"ai_powered":ai.get("ai_powered",False),"video_title":title,"tiktok_score":max(score["tiktok_score"],ai.get("score_viral",0)),"grade":score["grade"],"grade_color":score["grade_color"],"tt_details":score["details"],"has_subtitles":False})

        if not exported: raise ValueError("Export échoué. Réessaie avec une autre vidéo.")
        premium=[c for c in exported if c.get("grade") in {"A","B"}]
        if premium:
            for idx,c in enumerate(premium): c["rank"]=idx+1
            exported=premium
        # ✅ Tri par score_ia (Claude) en priorité, puis tiktok_score heuristique
        exported.sort(key=lambda c: (c.get("score_ia",0), c.get("tiktok_score",0)), reverse=True)
        # ✅ Compteur coût IA
        ai_clips = sum(1 for c in exported if c.get("ai_powered"))
        if ai_clips: _add_cost(ai_clips)

        zip_path=job_dir/"tous_les_clips.zip"
        def _make_zip():
            with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_STORED) as zf:
                for c in exported:
                    p=job_dir/c["filename"]
                    if p.exists(): zf.write(p,c["filename"])
        await loop.run_in_executor(executor,_make_zip)
        zip_url=f"/outputs/{job_id}/tous_les_clips.zip" if zip_path.exists() else None
        ai_count=sum(1 for c in exported if c.get("ai_powered"))
        set_job(job_id,status="done",progress=100,clips=exported,zip_url=zip_url,message=f"🎉 {len(exported)} clips · {ai_count} analysés par Claude AI")
        await notify_ws(job_id)

    except Exception as e:
        set_job(job_id,status="error",message=f"❌ {str(e)}"); await notify_ws(job_id)
    finally:
        for p in [video_path]:
            try:
                if Path(p).exists(): os.remove(p)
            except: pass
        seg_dir=UPLOAD_DIR/f"{job_id}_segs"
        if seg_dir.exists(): shutil.rmtree(seg_dir,ignore_errors=True)

# ── Routes ────────────────────────────────────────────────────────────
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(ws: WebSocket, job_id: str):
    await ws.accept(); ws_connections[job_id].append(ws)
    try:
        if job_id in jobs: await ws.send_text(json.dumps(jobs[job_id],default=str))
        while True:
            try:
                data=await asyncio.wait_for(ws.receive_text(),timeout=30)
                if data=="ping": await ws.send_text(json.dumps({"ping":True}))
            except asyncio.TimeoutError: await ws.send_text(json.dumps({"ping":True}))
    except WebSocketDisconnect: pass
    finally:
        try: ws_connections[job_id].remove(ws)
        except ValueError: pass

@app.post("/api/upload")
async def upload(request: Request, background_tasks: BackgroundTasks):
    client_ip=request.headers.get("x-forwarded-for","127.0.0.1").split(",")[0].strip()
    active=[j for j in ip_jobs.get(client_ip,[]) if jobs.get(j,{}).get("status") in ("pending","processing","exporting")]
    if len(active)>=MAX_JOBS_PER_IP: raise HTTPException(429,"Trop de jobs en cours.")
    disposition=request.headers.get("content-disposition","")
    watermark=request.headers.get("x-watermark","")[:40]
    niche=request.headers.get("x-niche","mindset")
    use_whisper=request.headers.get("x-whisper","true").lower()=="true"
    chunk=await request.body()
    if len(chunk)<8: raise HTTPException(400,"Fichier trop petit.")
    if len(chunk)>MAX_FILE_SIZE: raise HTTPException(413,f"Fichier trop volumineux ({len(chunk)//(1024*1024)}MB). Maximum 2GB.")
    fn_match=re.search(r'filename="?([^";\n]+)"?',disposition)
    filename=sanitize_filename(fn_match.group(1) if fn_match else "video.mp4")
    ext=Path(filename).suffix.lower() or ".mp4"
    if ext not in (".mp4",".mov",".mkv",".avi",".webm",".m4v"): ext=".mp4"
    job_id=str(uuid.uuid4()); vid_path=str(UPLOAD_DIR/f"{job_id}{ext}")
    with open(vid_path,"wb") as f: f.write(chunk)
    settings={"watermark":watermark,"niche":niche,"whisper":use_whisper}
    jobs[job_id]={"status":"pending","progress":0,"message":"En attente...","clips":[],"job_id":job_id}
    jobs_created_at[job_id] = time.time()  # ✅ timestamp pour nettoyage 2h
    ip_jobs.setdefault(client_ip,[]).append(job_id)
    background_tasks.add_task(process_video_job,job_id,vid_path,filename,settings)
    return {"job_id":job_id}

@app.post("/api/youtube")
async def youtube_download(request: Request, background_tasks: BackgroundTasks):
    body=await request.json()
    url=body.get("url","").strip(); niche=body.get("niche","mindset")
    watermark=body.get("watermark","")[:40]; use_whisper=body.get("whisper",True)
    if not url or ("youtube.com" not in url and "youtu.be" not in url):
        raise HTTPException(400,"URL YouTube invalide")
    job_id=str(uuid.uuid4())
    jobs[job_id]={"status":"pending","progress":0,"message":"📥 Téléchargement YouTube...","clips":[],"job_id":job_id}
    jobs_created_at[job_id] = time.time()  # ✅ timestamp pour nettoyage 2h
    async def _yt_job():
        try:
            set_job(job_id,status="processing",progress=3,message="📥 Téléchargement YouTube..."); await notify_ws(job_id)
            video_path,yt_title,yt_dur=await _download_youtube(url,job_id)
            set_job(job_id,video_title=yt_title)
            settings={"watermark":watermark,"niche":niche,"whisper":use_whisper,"_known_duration":yt_dur}
            await process_video_job(job_id,video_path,yt_title+".mp4",settings)
        except Exception as e:
            set_job(job_id,status="error",message=f"❌ {str(e)}"); await notify_ws(job_id)
    background_tasks.add_task(_yt_job)
    return {"job_id":job_id}

@app.post("/api/auto/trends")
async def auto_trends(request: Request):
    """Étape 1 — Claude génère 6 tendances thématiques pour la niche"""
    body = await request.json()
    niche = body.get("niche", "mindset")
    trends = await _auto_get_trends(niche)
    return {"trends": trends, "niche": niche}

@app.post("/api/auto/search")
async def auto_search(request: Request, background_tasks: BackgroundTasks):
    """Étape 2 — yt-dlp cherche les vraies vidéos pour un thème choisi"""
    body = await request.json()
    theme = body.get("theme", "")
    niche = body.get("niche", "mindset")
    if not theme:
        raise HTTPException(400, "Thème requis")
    loop = asyncio.get_running_loop()
    videos = await loop.run_in_executor(auto_executor, _search_youtube_for_theme, theme, niche, 4)
    return {"videos": videos, "theme": theme}

@app.post("/api/auto")
async def auto_mode(request: Request):
    """Rétrocompat — retourne tendances directement"""
    body = await request.json()
    niche = body.get("niche", "mindset")
    trends = await _auto_get_trends(niche)
    return {"trends": trends, "niche": niche}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs: raise HTTPException(404,"Job introuvable")
    return jobs[job_id]

@app.get("/api/jobs")
async def list_jobs():
    done=[j for j in jobs.values() if j.get("status")=="done" and j.get("clips")]
    done.sort(key=lambda j: len(j.get("clips",[])),reverse=True)
    return done[:20]

# ── Nettoyage automatique toutes les 2h ─────────────────────────────
async def _cleanup_loop():
    """Supprime les clips + jobs de plus de 2h toutes les 30min."""
    while True:
        await asyncio.sleep(1800)  # vérifie toutes les 30min
        now = time.time()
        expired = [jid for jid, ts in list(jobs_created_at.items()) if now - ts > CLIP_TTL_SECONDS]
        for jid in expired:
            # Supprime les fichiers
            job_dir = OUTPUT_DIR / jid
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            # Supprime les uploads résiduels
            for f in UPLOAD_DIR.glob(f"{jid}*"):
                try: f.unlink()
                except: pass
            # Nettoie les dicts mémoire
            jobs.pop(jid, None)
            jobs_created_at.pop(jid, None)
            ws_connections.pop(jid, None)
        # Nettoie ip_jobs des références mortes
        for ip in list(ip_jobs.keys()):
            ip_jobs[ip] = [j for j in ip_jobs[ip] if j in jobs]
            if not ip_jobs[ip]: del ip_jobs[ip]
        if expired:
            print(f"🧹 Nettoyage 2h : {len(expired)} job(s) supprimé(s)")

@app.on_event("startup")
async def startup():
    asyncio.create_task(_cleanup_loop())

@app.get("/health")
async def health():
    has_whisper = get_whisper_model() is not None
    return {
        "status":    "ok",
        "whisper":   has_whisper,
        "claude":    bool(ANTHROPIC_API_KEY),
        "version":   "PRO v2.0",
        "cost":      _cost_data,
        "jobs_actifs": len([j for j in jobs.values() if j.get("status") in ("pending","processing","exporting")]),
    }

# ── Route regénération hook ─────────────────────────────────────────
@app.post("/api/regenerate-hook")
async def regenerate_hook(request: Request):
    """Regénère uniquement le hook + titre d'un clip — ~0.001€"""
    body = await request.json()
    transcript = body.get("transcript","")
    titre_actuel = body.get("titre","")
    niche = body.get("niche","mindset")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503,"Clé Claude API requise")
    system = f"Tu es expert TikTok viral FR niche {niche}. Réponds UNIQUEMENT en JSON valide sans markdown."
    user = f"""Regénère un NOUVEAU hook et titre différents de l'actuel pour ce clip TikTok.
Titre actuel (à NE PAS reproduire) : "{titre_actuel}"
Transcription : "{transcript[:400] or 'Non disponible'}"
JSON requis : {{"titre": "Nouveau titre max 55 chars", "hook": "Nouveau hook max 70 chars"}}"""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
                json={"model":"claude-haiku-4-5-20251001","max_tokens":150,"system":system,
                      "messages":[{"role":"user","content":user}]})
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r'^```json\s*','',raw); raw = re.sub(r'\s*```$','',raw)
        data = json.loads(raw)
        return {"titre": data.get("titre",""), "hook": data.get("hook","")}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Route coût IA ────────────────────────────────────────────────────
@app.get("/api/cost")
async def get_cost():
    return _cost_data

app.mount("/outputs",StaticFiles(directory="outputs"),name="outputs")
app.mount("/",StaticFiles(directory="static",html=True),name="static")
