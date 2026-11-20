#!/usr/bin/env python3
"""Analyze OCR raw data to detect countdown timer + active exercise per frame."""
import json, re, sys

WORK = "/Users/guo/Documents/deepseek-harness/YouTube Studio/workout_remaster"
raw = json.load(open(f"{WORK}/data/ocr_raw.json"))

# Action names (cleaned from OCR noise). Order as shown in menu.
ACTIONS = [
    (1, "死虫子", "Dead Bug"),
    (2, "坐姿摸膝", "Seated Knee Touch"),
    (3, "坐姿卷腹", "Seated Crunch"),
    (4, "两边摸脚", "Alternating Toe Touch"),
    (5, "仰卧抬腿", "Leg Raise"),
    (6, "仰卧交叉", "Cross Crunch"),
    (7, "俄罗斯转体", "Russian Twist"),
    (8, "两头起", "Double Leg Raise"),
    (9, "背部两头起", "Back Extension"),
    (10, "拉伸", "Stretch"),
]

# Timer region: y between 140 and 210
def get_timer_texts(frame):
    out = []
    for t in frame["texts"]:
        ys = [p[1] for p in t["box"]]
        avg_y = sum(ys)/len(ys)
        if 140 <= avg_y <= 210:
            out.append((t["text"], t["conf"], avg_y))
    return out

def get_menu_texts(frame):
    out = []
    for t in frame["texts"]:
        ys = [p[1] for p in t["box"]]
        avg_y = sum(ys)/len(ys)
        if avg_y < 120:
            out.append((t["text"], t["conf"], avg_y))
    return out

# Detect timer value per frame
timer_seq = []
for f in raw:
    sec = f["sec"]
    tt = get_timer_texts(f)
    # find best candidate: numeric-looking
    best = None
    for txt, conf, y in tt:
        m = re.search(r'(\d{1,2})\s*[:.]?\s*(\d{2})', txt)
        if m:
            mm, ss = int(m.group(1)), int(m.group(2))
            best = (mm*60+ss, txt, conf, y)
            break
    timer_seq.append((sec, best, tt))

# Print timer transitions
prev = None
for sec, best, tt in timer_seq:
    if best and best[0] != prev:
        print(f"sec={sec:4d} timer={best[0]:4d} ({best[0]//60:02d}:{best[0]%60:02d}) raw='{best[1]}' conf={best[2]:.2f}")
        prev = best[0]

json.dump(timer_seq, open(f"{WORK}/data/timer_seq.json","w"))