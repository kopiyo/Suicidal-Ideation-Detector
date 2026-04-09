"""
Test video download and transcription outside Streamlit.
Run: python test_video.py
"""
import subprocess, tempfile, os
from pathlib import Path

URL = "https://www.tiktok.com/@salvadorlerma/video/7616193635763162381?is_from_webapp=1&sender_device=pc"  # replace with your URL

print("Step 1 - Downloading audio...")
with tempfile.TemporaryDirectory() as tmpdir:
    out_template = os.path.join(tmpdir, "audio.%(ext)s")
    cmd = ["yt-dlp", "--extract-audio", "--audio-format", "mp3",
           "--audio-quality", "5", "--no-playlist", "--quiet",
           "-o", out_template, URL]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print(f"Download FAILED: {result.stderr.strip()}")
        raise SystemExit

    files = list(Path(tmpdir).glob("audio.*"))
    if not files:
        print("ERROR: No audio file found after download")
        raise SystemExit

    audio_path = str(files[0])
    size = os.path.getsize(audio_path)
    print(f"Downloaded: {audio_path} ({size} bytes)")

    print("\nStep 2 - Transcribing...")
    from faster_whisper import WhisperModel
    wm = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = wm.transcribe(audio_path, beam_size=3)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"Transcript ({len(text)} chars):\n{text[:300]}")

print("\nAll done.")
