#!/usr/bin/env python3
"""OCR all extracted frames with easyocr (ch_sim + en). Saves per-frame results."""
import os, sys, json, time, glob, re
os.environ['HOME'] = '/tmp'
import cv2
import easyocr

WORK = "/Users/guo/Documents/deepseek-harness/YouTube Studio/workout_remaster"
MODEL_DIR = f"{WORK}/assets/easyocr_models"
os.environ['EASYOCR_MODEL_DIR'] = MODEL_DIR

langs = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ch_sim', 'en']
reader = easyocr.Reader(langs, gpu=True, model_storage_directory=MODEL_DIR,
                       download_enabled=False)

frames = sorted(glob.glob(f"{WORK}/frames/frame_*.png"))
print(f"processing {len(frames)} frames with langs={langs}", flush=True)

out = []
for i, fp in enumerate(frames):
    idx = int(re.search(r'frame_(\d+)', fp).group(1))
    img = cv2.imread(fp)
    if img is None:
        continue
    try:
        res = reader.readtext(img, detail=1, paragraph=False)
    except Exception as e:
        print(f"ERR frame {idx}: {e}", flush=True)
        res = []
    texts = []
    for r in res:
        box, txt, conf = r[0], r[1], r[2]
        texts.append({"text": txt, "conf": round(float(conf), 3),
                      "box": [[round(float(x), 1) for x in p] for p in box]})
    out.append({"frame": idx, "sec": idx, "texts": texts})
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(frames)} done", flush=True)

with open(f"{WORK}/data/ocr_raw.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("SAVED", f"{WORK}/data/ocr_raw.json")