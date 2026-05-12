import os
import uuid
import shutil
import subprocess
import threading
import time
import logging
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template

import librosa
import soundfile as sf
import numpy as np
from static_ffmpeg import run as _static_ffmpeg_run
import nodejs_wheel as _nodejs_wheel

app = Flask(__name__)

FFMPEG_BIN, FFPROBE_BIN = _static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
FFMPEG_DIR = str(Path(FFMPEG_BIN).parent)
NODE_BIN = str(Path(_nodejs_wheel.executable.ROOT_DIR) / "bin" / "node")
# Make ffmpeg/ffprobe/node discoverable by child libraries (yt-dlp, etc.)
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + str(Path(NODE_BIN).parent) + os.pathsep + os.environ.get("PATH", "")

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("choralia")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 Mo
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "ogg", "m4a", "aac", "opus", "webm", "3gp", "weba"}

# MIME type -> extension mapping for mobile browsers that send files without proper extensions
MIME_TO_EXT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/ogg": "ogg",
    "audio/vorbis": "ogg",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/aac": "aac",
    "audio/opus": "opus",
    "audio/webm": "webm",
    "audio/3gpp": "3gp",
    "audio/3gpp2": "3gp",
    "video/webm": "webm",
    "video/3gpp": "3gp",
}

# Store job status: job_id -> {status, progress, result, error}
jobs: dict[str, dict] = {}

# GPU lock: serialize all model inference to avoid OOM with concurrent users
_gpu_lock = threading.Lock()

# Demucs model cache: model_name -> model
_demucs_models: dict[str, object] = {}

# Last usage timestamp for automatic model unloading (0 = never loaded)
_demucs_last_used: float = 0
_MODEL_IDLE_TIMEOUT = 600  # 10 minutes

_FILE_RETENTION_SECONDS = 60 * 60  # 1 hour: files in uploads/ and outputs/ untouched for this long are removed
_CLEANUP_INTERVAL_SECONDS = 60     # how often the background loop checks


def write_mp3(path: Path, audio: np.ndarray, sr: int, bitrate: str = "192k") -> None:
    """Write a numpy audio array as MP3 via ffmpeg.

    audio shape: (samples,) for mono or (samples, channels) for stereo
    (i.e. the same layout soundfile.write expects).
    """
    # Write a temporary WAV next to the target, then transcode with ffmpeg.
    # Going through a temp WAV is simpler/safer than piping raw PCM and keeps
    # this code agnostic to bit depth and channel layouts.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(path.parent)) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio, samplerate=sr)
        cmd = [
            FFMPEG_BIN, "-y", "-loglevel", "error",
            "-i", tmp_path,
            "-codec:a", "libmp3lame", "-b:a", bitrate,
            str(path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resolve_extension(file) -> str | None:
    """Determine file extension from filename or MIME type (for mobile uploads)."""
    filename = file.filename or ""
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            return ext
    # Fallback: try to determine from MIME type (mobile browsers often send this)
    mime = file.content_type or ""
    return MIME_TO_EXT.get(mime)


def cleanup_old_files(max_age_seconds: int = _FILE_RETENTION_SECONDS):
    """Remove files/folders in uploads and outputs whose mtime is older than max_age_seconds.

    For directories (e.g. outputs/<file_id>/ from a separation), we look at the most
    recent mtime among the directory itself and its contents — so a folder that contains
    a recently-created stem isn't deleted prematurely.
    """
    now = time.time()
    removed = 0
    for directory in (UPLOAD_DIR, OUTPUT_DIR):
        if not directory.exists():
            continue
        for item in directory.iterdir():
            try:
                if item.is_file():
                    age = now - item.stat().st_mtime
                    if age > max_age_seconds:
                        log.info("Nettoyage : suppression de %s (âge %.0fs)", item.name, age)
                        item.unlink(missing_ok=True)
                        removed += 1
                elif item.is_dir():
                    # Use the most recently modified item inside the dir as the freshness signal
                    latest = item.stat().st_mtime
                    for child in item.rglob("*"):
                        try:
                            latest = max(latest, child.stat().st_mtime)
                        except OSError:
                            pass
                    age = now - latest
                    if age > max_age_seconds:
                        log.info("Nettoyage : suppression du dossier %s (âge %.0fs)", item.name, age)
                        shutil.rmtree(item, ignore_errors=True)
                        removed += 1
            except OSError as e:
                log.warning("Nettoyage : impossible de traiter %s : %s", item, e)
    if removed:
        log.info("Nettoyage terminé : %d élément(s) supprimé(s)", removed)


# --- Automatic cleanup loops ---

def _cleanup_loop():
    """Background thread:
       1) deletes uploads/outputs files untouched for _FILE_RETENTION_SECONDS
       2) unloads Demucs models after _MODEL_IDLE_TIMEOUT seconds of inactivity
    """
    global _demucs_last_used

    while True:
        time.sleep(_CLEANUP_INTERVAL_SECONDS)
        now = time.time()

        try:
            cleanup_old_files()
        except Exception as e:
            log.error("Nettoyage périodique : erreur %s", e, exc_info=True)

        with _gpu_lock:
            if _demucs_models and _demucs_last_used and (now - _demucs_last_used) > _MODEL_IDLE_TIMEOUT:
                log.info("Déchargement des modèles Demucs (inactifs depuis %ds)", int(now - _demucs_last_used))
                _demucs_models.clear()
                _demucs_last_used = 0
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log.info("Cache GPU vidé")


_cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
_cleanup_thread.start()


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        log.warning("Upload sans fichier joint")
        return jsonify({"error": "Aucun fichier envoyé"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    ext = resolve_extension(file)
    if not ext:
        log.warning("Upload refusé : format non supporté (filename=%s, mime=%s)", file.filename, file.content_type)
        return jsonify({"error": "Format non supporté. Formats acceptés : MP3, WAV, FLAC, OGG, M4A, AAC, OPUS, WEBM"}), 400

    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}.{ext}"
    filepath = UPLOAD_DIR / safe_name
    file.save(str(filepath))

    size_mb = filepath.stat().st_size / (1024 * 1024)
    log.info("Upload OK : %s (%.1f Mo) → %s", file.filename, size_mb, safe_name)

    return jsonify({
        "file_id": file_id,
        "filename": file.filename or f"audio.{ext}",
        "stored_as": safe_name,
    })


_YT_URL_RE = r'^https?://(www\.|m\.)?(youtube\.com/watch\?v=|youtu\.be/|music\.youtube\.com/watch\?v=)[\w-]+'
_YT_VIDEO_ID_RE = r'^[\w-]{11}$'


@app.route("/search_youtube", methods=["POST"])
def search_youtube():
    import re
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Paramètre 'query' manquant"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Requête vide"}), 400
    if len(query) > 200:
        return jsonify({"error": "Requête trop longue"}), 400

    try:
        limit = int(data.get("limit", 10))
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(20, limit))

    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "noplaylist": True,
            "default_search": f"ytsearch{limit}",
            "js_runtimes": {"node": {"path": NODE_BIN}},
            "remote_components": ["ejs:github"],
        }
        t0 = time.time()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)

        entries = info.get("entries", []) if info else []
        results = []
        for e in entries:
            if not e:
                continue
            vid = e.get("id")
            if not vid:
                continue
            # Pick the best available thumbnail (highest resolution)
            thumb = None
            thumbs = e.get("thumbnails") or []
            if thumbs:
                thumb = thumbs[-1].get("url")
            if not thumb:
                thumb = f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"
            results.append({
                "video_id": vid,
                "title": e.get("title") or "(sans titre)",
                "channel": e.get("channel") or e.get("uploader") or "",
                "duration": e.get("duration"),  # seconds, may be None
                "thumbnail": thumb,
                "url": f"https://www.youtube.com/watch?v={vid}",
            })

        log.info("Recherche YouTube « %s » : %d résultats en %.1fs", query, len(results), time.time() - t0)
        return jsonify({"results": results})

    except Exception as e:
        log.error("Recherche YouTube échouée (« %s ») : %s", query, e)
        return jsonify({"error": f"Erreur lors de la recherche : {e}"}), 500


@app.route("/import_youtube", methods=["POST"])
def import_youtube():
    import re
    data = request.get_json()
    if not data:
        return jsonify({"error": "Données manquantes"}), 400

    url = (data.get("url") or "").strip()
    video_id = (data.get("video_id") or "").strip()

    if video_id:
        if not re.match(_YT_VIDEO_ID_RE, video_id):
            return jsonify({"error": "video_id invalide"}), 400
        url = f"https://www.youtube.com/watch?v={video_id}"
    elif url:
        if not re.match(_YT_URL_RE, url):
            return jsonify({"error": "URL YouTube invalide"}), 400
    else:
        return jsonify({"error": "Paramètre 'url' ou 'video_id' manquant"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "progress": "Démarrage du téléchargement..."}

    log.info("Import YouTube démarré : %s (job=%s)", url, job_id)
    thread = threading.Thread(target=_run_youtube_import, args=(job_id, url))
    thread.start()

    return jsonify({"job_id": job_id})


def _run_youtube_import(job_id: str, url: str):
    try:
        import yt_dlp

        file_id = str(uuid.uuid4())
        output_path = str(UPLOAD_DIR / file_id)

        jobs[job_id]["progress"] = "Téléchargement depuis YouTube..."

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path + ".%(ext)s",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "noplaylist": True,
            "quiet": True,
            "ffmpeg_location": FFMPEG_DIR,
            "js_runtimes": {"node": {"path": NODE_BIN}},
            "remote_components": ["ejs:github"],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "audio")
            duration = info.get("duration", 0)

        stored_as = f"{file_id}.mp3"
        final_path = UPLOAD_DIR / stored_as

        # yt-dlp may produce the file with the expected name
        if not final_path.exists():
            # Try to find the downloaded file
            for f in UPLOAD_DIR.iterdir():
                if f.stem == file_id and f.suffix != ".part":
                    f.rename(final_path)
                    break

        if not final_path.exists():
            log.error("Import YouTube échoué : fichier introuvable après téléchargement (job=%s)", job_id)
            jobs[job_id] = {"status": "error", "error": "Échec du téléchargement"}
            return

        size_mb = final_path.stat().st_size / (1024 * 1024)
        log.info("Import YouTube OK : « %s » (durée=%ds, %.1f Mo) → %s", title, duration, size_mb, stored_as)

        jobs[job_id] = {
            "status": "done",
            "file_id": file_id,
            "filename": f"{title}.mp3",
            "stored_as": stored_as,
        }

    except Exception as e:
        log.error("Import YouTube échoué (job=%s) : %s", job_id, e)
        jobs[job_id] = {"status": "error", "error": str(e)}


@app.route("/separate", methods=["POST"])
def separate():
    data = request.get_json()
    if not data or "stored_as" not in data:
        return jsonify({"error": "Paramètre 'stored_as' manquant"}), 400

    stored_as = data["stored_as"]
    model_name = data.get("model", "htdemucs_6s")
    if model_name not in ("htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx_extra"):
        return jsonify({"error": "Modèle invalide"}), 400

    filepath = UPLOAD_DIR / stored_as
    if not filepath.exists():
        log.warning("Séparation : fichier introuvable %s", stored_as)
        return jsonify({"error": "Fichier introuvable"}), 404

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "progress": "Démarrage de la séparation..."}

    size_mb = filepath.stat().st_size / (1024 * 1024)
    log.info("Séparation démarrée : %s (%.1f Mo), modèle=%s (job=%s)", stored_as, size_mb, model_name, job_id)

    thread = threading.Thread(target=_run_separation, args=(job_id, filepath, stored_as, model_name))
    thread.start()

    return jsonify({"job_id": job_id})


def _get_demucs_model(model_name: str):
    """Lazy-load and cache Demucs models."""
    global _demucs_last_used
    if model_name not in _demucs_models:
        import torch
        from demucs.pretrained import get_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Chargement du modèle Demucs « %s » sur %s...", model_name, device)
        t0 = time.time()
        model = get_model(model_name)
        model.to(device)
        model.eval()
        _demucs_models[model_name] = model
        log.info("Modèle « %s » chargé en %.1fs", model_name, time.time() - t0)
    else:
        log.info("Modèle « %s » déjà en cache", model_name)
    _demucs_last_used = time.time()
    return _demucs_models[model_name]


def _run_separation(job_id: str, filepath: Path, stored_as: str, model_name: str = "htdemucs"):
    t_start = time.time()
    try:
        import torch
        from demucs.apply import apply_model

        jobs[job_id]["progress"] = "Chargement du modèle Demucs..."

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _get_demucs_model(model_name)

        jobs[job_id]["progress"] = "Chargement du fichier audio..."
        log.info("[job=%s] Chargement audio : %s", job_id[:8], filepath.name)

        # Load audio with librosa (mono=False to keep channels)
        y, sr = librosa.load(str(filepath), sr=model.samplerate, mono=False)
        duration_sec = y.shape[-1] / sr
        log.info("[job=%s] Audio chargé : %.1fs, sr=%d, channels=%s", job_id[:8], duration_sec, sr, y.shape[0] if y.ndim > 1 else 1)

        # y shape: (channels, samples) or (samples,) if mono
        if y.ndim == 1:
            y = np.stack([y, y])  # mono -> stereo
        elif y.shape[0] > 2:
            y = y[:2]
        wav = torch.tensor(y, dtype=torch.float32).to(device)
        # Add batch dimension: (batch, channels, samples)
        wav = wav.unsqueeze(0)

        if _gpu_lock.locked():
            log.info("[job=%s] En file d'attente GPU...", job_id[:8])
            jobs[job_id]["progress"] = "En file d'attente (un autre traitement est en cours)..."
        jobs[job_id]["progress"] = "Séparation en cours (cela peut prendre quelques minutes)..."

        with _gpu_lock:
            t_infer = time.time()
            log.info("[job=%s] Inférence Demucs démarrée", job_id[:8])
            with torch.no_grad():
                sources = apply_model(model, wav, device=device)
            # sources shape: (batch, num_sources, channels, samples)
            sources = sources[0]  # remove batch dim
            log.info("[job=%s] Inférence terminée en %.1fs", job_id[:8], time.time() - t_infer)

        file_id = stored_as.rsplit(".", 1)[0]
        output_subdir = OUTPUT_DIR / file_id
        output_subdir.mkdir(exist_ok=True)

        tracks = {}
        for i, stem_name in enumerate(model.sources):
            stem_audio = sources[i].cpu().numpy()
            # stem_audio shape: (channels, samples)
            out_path = output_subdir / f"{stem_name}.mp3"
            write_mp3(out_path, stem_audio.T, sr)
            tracks[stem_name] = f"{file_id}/{stem_name}.mp3"

        # Create instrumental mix (everything except vocals)
        instrumental_parts = []
        for i, stem_name in enumerate(model.sources):
            if stem_name != "vocals":
                instrumental_parts.append(sources[i].cpu().numpy())

        if instrumental_parts:
            instrumental = sum(instrumental_parts)
            out_path = output_subdir / "instrumental.mp3"
            write_mp3(out_path, instrumental.T, sr)
            tracks["instrumental"] = f"{file_id}/instrumental.mp3"

        elapsed = time.time() - t_start
        log.info("[job=%s] Séparation terminée : %d pistes en %.1fs (%s)", job_id[:8], len(tracks), elapsed, ", ".join(tracks.keys()))
        jobs[job_id] = {"status": "done", "tracks": tracks}

    except Exception as e:
        log.error("[job=%s] Séparation échouée après %.1fs : %s", job_id[:8], time.time() - t_start, e, exc_info=True)
        jobs[job_id] = {"status": "error", "error": str(e)}


@app.route("/job_status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable"}), 404
    return jsonify(job)


@app.route("/transpose", methods=["POST"])
def transpose():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Données manquantes"}), 400

    source = data.get("source")  # relative path under outputs/ or stored_as under uploads/
    semitones = data.get("semitones", 0)

    if not source:
        return jsonify({"error": "Paramètre 'source' manquant"}), 400

    try:
        semitones = int(semitones)
    except (TypeError, ValueError):
        return jsonify({"error": "Le nombre de demi-tons doit être un entier"}), 400

    if not -12 <= semitones <= 12:
        return jsonify({"error": "Le nombre de demi-tons doit être entre -12 et +12"}), 400

    if semitones == 0:
        return jsonify({"error": "Choisissez un nombre de demi-tons différent de 0"}), 400

    # Try to find the source file in outputs/ first, then uploads/
    source_path = OUTPUT_DIR / source
    if not source_path.exists():
        source_path = UPLOAD_DIR / source
    if not source_path.exists():
        return jsonify({"error": "Fichier source introuvable"}), 404

    # Ensure the path doesn't escape our directories
    try:
        source_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        try:
            source_path.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            return jsonify({"error": "Chemin de fichier invalide"}), 400

    log.info("Transposition : source=%s, demi-tons=%+d", source, semitones)

    try:
        t0 = time.time()
        y, sr = librosa.load(str(source_path), sr=None)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

        sign = "plus" if semitones > 0 else "minus"
        source_stem = Path(source).stem
        out_name = f"{source_stem}_transposed_{sign}{abs(semitones)}.mp3"

        # Put transposed file next to the source if it's in a subdir, otherwise in outputs/
        if "/" in source:
            subdir = source.split("/")[0]
            out_dir = OUTPUT_DIR / subdir
        else:
            out_dir = OUTPUT_DIR
        out_dir.mkdir(exist_ok=True)

        out_path = out_dir / out_name
        # write_mp3 expects (samples,) or (samples, channels)
        if y_shifted.ndim > 1:
            write_mp3(out_path, y_shifted.T, sr)
        else:
            write_mp3(out_path, y_shifted, sr)

        log.info("Transposition OK : %s en %.1fs", out_name, time.time() - t0)

        relative = str(out_path.relative_to(OUTPUT_DIR))
        return jsonify({"transposed_file": relative, "filename": out_name})

    except Exception as e:
        log.error("Transposition échouée (source=%s) : %s", source, e, exc_info=True)
        return jsonify({"error": f"Erreur lors de la transposition : {e}"}), 500


@app.route("/mix", methods=["POST"])
def mix():
    data = request.get_json()
    if not data or "tracks" not in data:
        return jsonify({"error": "Paramètre 'tracks' manquant"}), 400

    track_paths = data["tracks"]
    if not isinstance(track_paths, list) or len(track_paths) < 2:
        return jsonify({"error": "Il faut au moins 2 pistes à assembler"}), 400

    # Validate all tracks exist and belong to the same file_id
    file_ids = set()
    resolved = []
    for rel_path in track_paths:
        full_path = OUTPUT_DIR / rel_path
        if not full_path.exists():
            return jsonify({"error": f"Piste introuvable : {rel_path}"}), 404
        try:
            full_path.resolve().relative_to(OUTPUT_DIR.resolve())
        except ValueError:
            return jsonify({"error": "Chemin de fichier invalide"}), 400
        resolved.append(full_path)
        file_ids.add(rel_path.split("/")[0])

    if len(file_ids) != 1:
        return jsonify({"error": "Toutes les pistes doivent provenir de la même séparation"}), 400

    file_id = file_ids.pop()
    log.info("Mix demandé : %d pistes de %s (%s)", len(resolved), file_id, ", ".join(p.stem for p in resolved))

    try:
        t0 = time.time()

        # Load all tracks
        signals = []
        sr_out = None
        for p in resolved:
            y, sr = librosa.load(str(p), sr=None, mono=False)
            if sr_out is None:
                sr_out = sr
            signals.append(y)

        # Sum all signals (same shape from Demucs)
        mix_signal = sum(signals)

        # Save
        short_id = str(uuid.uuid4())[:8]
        out_dir = OUTPUT_DIR / file_id
        out_dir.mkdir(exist_ok=True)
        out_name = f"mix_{short_id}.mp3"
        out_path = out_dir / out_name

        if mix_signal.ndim > 1:
            write_mp3(out_path, mix_signal.T, sr_out)
        else:
            write_mp3(out_path, mix_signal, sr_out)

        elapsed = time.time() - t0
        size_mb = out_path.stat().st_size / (1024 * 1024)
        relative = f"{file_id}/{out_name}"
        log.info("Mix OK : %s (%.1f Mo) en %.1fs — pistes : %s", out_name, size_mb, elapsed, ", ".join(p.stem for p in resolved))

        return jsonify({"mix_file": relative, "filename": out_name})

    except Exception as e:
        log.error("Mix échoué : %s", e, exc_info=True)
        return jsonify({"error": f"Erreur lors de l'assemblage : {e}"}), 500


@app.route("/download/<path:filename>")
def download(filename):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        log.warning("Téléchargement : fichier introuvable %s", filename)
        return jsonify({"error": "Fichier introuvable"}), 404

    # Security: ensure file is within allowed directories
    try:
        filepath.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        try:
            filepath.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            log.warning("Téléchargement : accès interdit %s", filename)
            return jsonify({"error": "Accès interdit"}), 403

    size_mb = filepath.stat().st_size / (1024 * 1024)
    log.info("Téléchargement : %s (%.1f Mo)", filename, size_mb)
    return send_file(str(filepath), as_attachment=True)


if __name__ == "__main__":
    import argparse
    import subprocess
    import signal
    import atexit

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tunnel", action="store_true", help="Disable the Cloudflare tunnel (local-only mode)")
    args = parser.parse_args()
    tunnel_enabled = not args.no_tunnel

    tunnel_proc = None

    if tunnel_enabled:
        cloudflared_bin = BASE_DIR / "cloudflared"
        if cloudflared_bin.exists():
            print(" * Lancement du tunnel Cloudflare...")
            tunnel_proc = subprocess.Popen(
                [str(cloudflared_bin), "tunnel", "run", "choralia"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            def _stop_tunnel():
                if tunnel_proc and tunnel_proc.poll() is None:
                    print("\n * Arrêt du tunnel Cloudflare...")
                    tunnel_proc.terminate()
                    try:
                        tunnel_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        tunnel_proc.kill()

            atexit.register(_stop_tunnel)
            signal.signal(signal.SIGTERM, lambda *_: (_stop_tunnel(), exit(0)))

            # Attendre que le tunnel soit connecté (max 15s)
            import select
            deadline = time.time() + 15
            connected = False
            while time.time() < deadline:
                ready, _, _ = select.select([tunnel_proc.stderr], [], [], 0.5)
                if ready:
                    line = tunnel_proc.stderr.readline().decode("utf-8", errors="ignore")
                    if "Registered tunnel connection" in line or "Connection registered" in line:
                        connected = True
                        break
                if tunnel_proc.poll() is not None:
                    break

            if connected:
                print(" * Tunnel Cloudflare connecté !")
            else:
                print(" * Tunnel Cloudflare en cours de connexion...")
            print(" * Accès public → https://choralia.cedricgillot.fr")
        else:
            print(" * ATTENTION : cloudflared introuvable, tunnel non démarré")

    from waitress import serve

    log.info("Démarrage de Choralia (production, tunnel=%s)", tunnel_enabled)
    serve(app, host="0.0.0.0", port=5000, threads=8)
