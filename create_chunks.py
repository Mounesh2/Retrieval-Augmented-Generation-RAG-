import json
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent

TRANSCRIPT_DIR = BASE_DIR / "transcripts"
OUTPUT_DIR = BASE_DIR / "chunks"

OUTPUT_DIR.mkdir(exist_ok=True)

MAX_WORDS = 120
OVERLAP_WORDS = 20


# Get all transcript JSON files
files = sorted(
    TRANSCRIPT_DIR.glob("video_*.json"),
    key=lambda x: int(
        re.search(r"video_(\d+)", x.name).group(1)
    )
)

print(f"Found {len(files)} transcript files")


all_chunks = []


# Process every video
for file in files:

    print(f"\nProcessing: {file.name}")

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_number = data["video"]
    course = data["course"]
    source_file = data["source_file"]

    segments = data["segments"]

    current_text = []
    current_start = None
    current_end = None

    chunk_number = 1

    for segment in segments:

        text = segment["text"].strip()

        if not text:
            continue

        if current_start is None:
            current_start = segment["start"]

        current_text.append(text)
        current_end = segment["end"]

        word_count = len(
            " ".join(current_text).split()
        )

        if word_count >= MAX_WORDS:

            chunk_text = " ".join(current_text)

            chunk = {
                "chunk_id": (
                    f"video_{video_number:02d}"
                    f"_chunk_{chunk_number:03d}"
                ),
                "video": video_number,
                "course": course,
                "source_file": source_file,
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "text": chunk_text
            }

            all_chunks.append(chunk)

            print(
                f"  Chunk {chunk_number}: "
                f"{current_start:.2f}s -> "
                f"{current_end:.2f}s"
            )

            # Overlap
            words = chunk_text.split()

            current_text = words[-OVERLAP_WORDS:]

            current_start = current_end

            chunk_number += 1


    # Save remaining text
    if current_text:

        chunk_text = " ".join(current_text)

        chunk = {
            "chunk_id": (
                f"video_{video_number:02d}"
                f"_chunk_{chunk_number:03d}"
            ),
            "video": video_number,
            "course": course,
            "source_file": source_file,
            "start": round(current_start, 2),
            "end": round(current_end, 2),
            "text": chunk_text
        }

        all_chunks.append(chunk)


# Save all chunks
output_file = OUTPUT_DIR / "all_chunks.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n" + "=" * 60)
print("CHUNKING COMPLETE")
print("=" * 60)

print(f"Total chunks: {len(all_chunks)}")
print(f"Saved to: {output_file}")