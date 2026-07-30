import os
import uuid
import wave
import subprocess

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import yt_dlp
import whisper

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

model = None

def get_model():
    global model
    if model is None:
        print("Loading Whisper model, please wait...")
        model = whisper.load_model("tiny")
        print("Whisper Model Loaded Successfully!")
    return model

QUALITY_MAP = {
    "720p": "1280:720",
    "1080p": "1920:1080",
    "4k": "3840:2160",
}

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MAX_SHORT_DURATION = 45  # seconds, auto highlight window length for long source videos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ffmpeg_path_escape(path):
    """Make an absolute path safe to pass inside an ffmpeg -vf 'subtitles=...' filter."""
    abs_path = os.path.abspath(path)
    abs_path = abs_path.replace("\\", "/")
    abs_path = abs_path.replace(":", "\\:")
    return abs_path


def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def write_srt(segments, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")


def get_duration_seconds(video_path):
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 1. Auto highlight selection - finds the most "energetic" (loudest, most
#    likely engaging) window in the source video instead of always taking
#    the beginning.
# ---------------------------------------------------------------------------

def find_highlight_start(raw_path, target_duration, total_duration):
    if total_duration <= target_duration:
        return 0.0

    audio_tmp = raw_path + "_analysis.wav"
    try:
        extract_cmd = [
            "ffmpeg", "-y", "-i", raw_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", audio_tmp,
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)

        with wave.open(audio_tmp, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_samples = wf.readframes(n_frames)

        samples = np.frombuffer(raw_samples, dtype=np.int16).astype(np.float32)
        bucket_size = sample_rate  # 1-second buckets
        n_buckets = max(1, len(samples) // bucket_size)

        energies = np.zeros(n_buckets)
        for i in range(n_buckets):
            chunk = samples[i * bucket_size:(i + 1) * bucket_size]
            energies[i] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0.0

        window = int(target_duration)
        if n_buckets <= window:
            return 0.0

        window_sums = np.convolve(energies, np.ones(window), mode="valid")
        best_start = int(np.argmax(window_sums))
        return float(best_start)

    except Exception:
        # If audio analysis fails for any reason, safely fall back to the start
        return 0.0
    finally:
        if os.path.exists(audio_tmp):
            try:
                os.remove(audio_tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 2. Face-tracking dynamic crop - instead of a fixed center crop, the crop
#    window follows the detected face horizontally, frame by frame.
# ---------------------------------------------------------------------------

def dynamic_face_crop(input_path, output_video_only_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Work out the 9:16 crop box that fits inside the source frame
    crop_w = int(orig_h * 9 / 16)
    crop_h = orig_h
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(orig_w * 16 / 9)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_only_path, fourcc, fps, (crop_w, crop_h))

    smoothed_center_x = orig_w / 2.0
    smoothing_alpha = 0.15
    detect_every_n = 3
    frame_idx = 0
    last_detected_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % detect_every_n == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces):
                # Track the largest detected face
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest
                last_detected_center = fx + fw / 2.0

        target_center = last_detected_center if last_detected_center is not None else orig_w / 2.0
        smoothed_center_x = (
            smoothing_alpha * target_center + (1 - smoothing_alpha) * smoothed_center_x
        )

        crop_x = int(smoothed_center_x - crop_w / 2)
        crop_x = max(0, min(crop_x, orig_w - crop_w))
        crop_y = int((orig_h - crop_h) / 2)

        cropped_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        writer.write(cropped_frame)
        frame_idx += 1

    cap.release()
    writer.release()


# ---------------------------------------------------------------------------
# 3. Adaptive color grading - measures average brightness and tunes the
#    grade instead of applying one fixed look to every video.
# ---------------------------------------------------------------------------

def measure_brightness(video_path, samples=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    brightness_values = []

    for i in range(samples):
        frame_pos = int((i / samples) * total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(float(np.mean(gray)))

    cap.release()
    if not brightness_values:
        return 128.0  # neutral fallback
    return float(np.mean(brightness_values))


def adaptive_grade_filter(brightness):
    # brightness is 0-255; 128 is neutral midtone
    if brightness < 90:
        brightness_adjust = 0.06   # brighten dark footage
        contrast = 1.10
        saturation = 1.15
    elif brightness > 170:
        brightness_adjust = -0.03  # tone down overexposed footage
        contrast = 1.05
        saturation = 1.12
    else:
        brightness_adjust = 0.02
        contrast = 1.08
        saturation = 1.18

    return f"eq=contrast={contrast}:saturation={saturation}:brightness={brightness_adjust},vignette=PI/5"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-video", methods=["POST"])
def process_video():
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    language = data.get("language", "auto")
    quality = data.get("quality", "1080p")

    if not url:
        return jsonify({"error": "YouTube URL ivvandi"}), 400

    job_id = uuid.uuid4().hex[:10]
    raw_path = os.path.join(DOWNLOAD_DIR, f"{job_id}_raw.mp4")
    trimmed_path = os.path.join(DOWNLOAD_DIR, f"{job_id}_trimmed.mp4")
    crop_noaudio_path = os.path.join(DOWNLOAD_DIR, f"{job_id}_cropvid.mp4")
    cropped_path = os.path.join(DOWNLOAD_DIR, f"{job_id}_cropped.mp4")
    srt_path = os.path.join(DOWNLOAD_DIR, f"{job_id}.srt")
    final_path = os.path.join(DOWNLOAD_DIR, f"{job_id}_final.mp4")

    cleanup_paths = [raw_path, trimmed_path, crop_noaudio_path, cropped_path, srt_path]

    try:
        # 1. Download video
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "outtmpl": raw_path,
            "quiet": True,
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(raw_path):
            return jsonify({"error": "Video download avvaledhu, URL check cheyandi"}), 500

        total_duration = get_duration_seconds(raw_path) or 0
        target_duration = min(MAX_SHORT_DURATION, total_duration) if total_duration else MAX_SHORT_DURATION

        # 2. Auto highlight window (skips straight to the most energetic part)
        highlight_start = find_highlight_start(raw_path, target_duration, total_duration)

        trim_cmd = [
            "ffmpeg", "-y",
            "-ss", str(highlight_start),
            "-i", raw_path,
            "-t", str(target_duration),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            trimmed_path,
        ]
        subprocess.run(trim_cmd, check=True, capture_output=True)

        # 3. Face-tracking dynamic crop (video only)
        dynamic_face_crop(trimmed_path, crop_noaudio_path)

        # 4. Mux the original trimmed audio back onto the cropped video
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", crop_noaudio_path,
            "-i", trimmed_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac",
            "-shortest",
            cropped_path,
        ]
        subprocess.run(mux_cmd, check=True, capture_output=True)

        # 5. Transcribe with Whisper
        whisper_kwargs = {"fp16": False}
        if language in ("te", "en"):
        whisper_kwargs["language"] = language
        result = get_model().transcribe(cropped_path, **whisper_kwargs)
        detected_lang = result.get("language", language)
        font_name = "Noto Sans Telugu" if detected_lang == "te" else "Arial"

        # 6. Adaptive color grade based on measured brightness
        brightness = measure_brightness(cropped_path)
        color_grade_filter = adaptive_grade_filter(brightness)

        # 7. Final render: color grade + subtitles + scale + fade transitions
        escaped_srt = ffmpeg_path_escape(srt_path)
        scale = QUALITY_MAP.get(quality, QUALITY_MAP["1080p"])

        subtitle_filter = (
            f"subtitles='{escaped_srt}':"
            f"force_style='FontName={font_name},FontSize=16,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"BorderStyle=1,Outline=2,Shadow=0,Alignment=2,"
            f"MarginV=40,MarginL=30,MarginR=30,WrapStyle=2'"
        )

        clip_duration = get_duration_seconds(cropped_path) or target_duration
        fade_duration = 0.5
        fade_out_start = max(0, clip_duration - fade_duration)
        fade_filter = (
            f"fade=t=in:st=0:d={fade_duration},"
            f"fade=t=out:st={fade_out_start:.2f}:d={fade_duration}"
        )

        full_filter = ",".join([color_grade_filter, subtitle_filter, f"scale={scale}", fade_filter])

        burn_cmd = [
            "ffmpeg", "-y", "-i", cropped_path,
            "-vf", full_filter,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            final_path,
        ]
        burn_result = subprocess.run(burn_cmd, capture_output=True, text=True)

        if burn_result.returncode != 0 or not os.path.exists(final_path):
            return jsonify({
                "error": "Video render fail ayindi: " + burn_result.stderr[-500:]
            }), 500

        return jsonify({
            "message": (
                f"Short ready! Auto highlight from {highlight_start:.0f}s, "
                f"face-tracked crop, adaptive grade, {detected_lang.upper()} subtitles, {quality}."
            ),
            "options": [
                {"quality": quality, "url": f"/downloads/{job_id}_final.mp4"}
            ]
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg step fail ayindi: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.route("/downloads/<path:filename>")
def serve_download(filename):
    return send_from_directory(DOWNLOAD_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
