import os

# Fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import whisper
import json
from pathlib import Path


# --------------------------------
# 1. Paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "audios"

OUTPUT_DIR = BASE_DIR / "transcripts"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------
# 2. Load Whisper
# --------------------------------

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Whisper loaded successfully!")


# --------------------------------
# 3. Get all MP3 files
# --------------------------------

audio_files = sorted(
    AUDIO_DIR.glob("*.mp3"),
    key=lambda file: int(file.name.split("_")[0])
)

print(f"Found {len(audio_files)} MP3 files")


# --------------------------------
# 4. Convert every MP3 to text
# --------------------------------

for number, audio_file in enumerate(audio_files, start=1):

    print("\n" + "=" * 70)
    print(f"Processing {number}/{len(audio_files)}")
    print(audio_file.name)
    print("=" * 70)

    result = model.transcribe(
        str(audio_file),
        language="hi",
        task="translate"
    )

    # --------------------------------
    # 5. Save text
    # --------------------------------

    video_number = int(audio_file.name.split("_")[0])

    output_file = OUTPUT_DIR / f"video_{video_number:02d}.json"

    data = {
        "video": video_number,
        "course": "Sigma Web Development",
        "source_file": audio_file.name,
        "language": result.get("language"),
        "text": result["text"],
        "segments": [
            {
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip()
            }
            for segment in result["segments"]
        ]
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Saved:", output_file)


print("\n")
print("=" * 70)
print("ALL MP3 FILES CONVERTED TO TEXT!")
print("=" * 70)