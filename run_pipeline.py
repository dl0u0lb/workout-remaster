#!/usr/bin/env python3
"""
Badminton & General Fitness Workout Video Remaster Pipeline
(Dynamic Motion Envelope Stage + Adaptive Same-Color Blend-In + English HUD)

100% Data-Driven & Dynamically Adaptive Architecture:
- Dynamically loads timeline configuration from `--timeline <json>`
- Computes global motion envelope and stage padding dynamically
- Samples edge textures dynamically from the stage borders
- Cleans original video overlays based on clean_regions in timeline or auto defaults
"""
import os
import sys
import json
import time
import math
import glob
import wave
import shutil
import struct
import asyncio
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import edge_tts

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==============================================================================
# Global Defaults & Configuration
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TIMELINE = SCRIPT_DIR / "timeline.json"
DEFAULT_TEST_DIR = SCRIPT_DIR / "test"
DEFAULT_OUT_DIR = SCRIPT_DIR / "output"
DEFAULT_ASSETS_DIR = SCRIPT_DIR / "assets"
DEFAULT_TMP_DIR = SCRIPT_DIR / "tmp"

SAM2_CKPT = "/Users/guo/models/sam2.1_hiera_small.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
FONT_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"
TTS_VOICE = "en-US-AndrewNeural"

TARGET_W, TARGET_H = 1920, 1080
SAMPLE_RATE = 44100


# ==============================================================================
# 1. SAM 2 Dynamic Motion Envelope Scanner
# ==============================================================================
def compute_motion_envelope(video_path: Path, timeline_data: dict, work_dir: Path) -> dict:
    """
    Dynamically scan 40+ keyframes across the video using SAM 2 to compute the
    global union bounding box of the athlete and mat/equipment, then expand by
    adaptive safety margins to form the Core Stage rectangle.
    """
    cache_path = work_dir / "tmp" / "motion_envelope.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                env = json.load(f)
            if "stage_src" in env and "stage_target" in env:
                print(
                    f"[MotionEnvelope] Loaded cached envelope ({env.get('num_keyframes_scanned', 0)} keyframes): "
                    f"src={env['stage_src']} -> target={env['stage_target']}",
                    flush=True,
                )
                return env
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    total_dur = total_frames / fps
    # Uniformly sample 45 keyframe timestamps across active duration (skip extreme fade out if any)
    active_end = total_dur
    if "outro" in timeline_data and "start_sec" in timeline_data["outro"]:
        active_end = float(timeline_data["outro"]["start_sec"]) + min(10.0, float(timeline_data["outro"].get("duration_sec", 10.0)))

    sample_sec = np.linspace(1.0, max(2.0, active_end - 1.0), 45)
    # Add key timestamps from exercises if available
    for b in timeline_data.get("blocks", []):
        s = float(b["start_sec"])
        d = float(b.get("duration_sec", 20.0))
        sample_sec = np.append(sample_sec, [s + 0.3 * d, s + 0.7 * d])
    sample_sec = np.unique(np.clip(sample_sec, 0.5, total_dur - 0.5))

    clean_cfg = timeline_data.get("clean_regions", {})
    top_y_max = clean_cfg.get("top_bar", {}).get("y_max", int(src_h * 0.38))
    bot_y_min = clean_cfg.get("bottom_bar", {}).get("y_min", int(src_h * 0.825))

    print(f"[SAM 2] Dynamically scanning {len(sample_sec)} keyframes ({src_w}x{src_h}) to find Motion Envelope...", flush=True)
    sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device="mps")
    predictor = SAM2ImagePredictor(sam2_model)

    gx0, gy0, gx1, gy1 = src_w, src_h, 0, 0
    cap = cv2.VideoCapture(str(video_path))

    with torch.inference_mode(), torch.autocast("mps", dtype=torch.float16):
        for s in sample_sec:
            f_idx = int(round(s * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_proc = frame.copy()
            frame_proc[:top_y_max, :] = 40
            frame_proc[bot_y_min:, :] = 40

            predictor.set_image(cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB))
            # Dynamic prompt box centered around lower 60% of frame
            p_box = np.array([int(src_w * 0.2), top_y_max, int(src_w * 0.8), bot_y_min], dtype=np.float32)
            pts = np.array([
                [int(src_w * 0.5), int(src_h * 0.65)],
                [int(src_w * 0.4), int(src_h * 0.60)],
                [int(src_w * 0.6), int(src_h * 0.60)],
            ], dtype=np.float32)

            masks, _, _ = predictor.predict(
                point_coords=pts,
                point_labels=np.array([1, 1, 1], dtype=np.int32),
                box=p_box,
                multimask_output=False,
            )
            m = masks[0] > 0
            m[:top_y_max, :] = False
            m[bot_y_min:, :] = False
            m[:, :int(src_w * 0.15)] = False
            m[:, int(src_w * 0.85):] = False

            m_u8 = (m.astype(np.uint8)) * 255
            num_l, labels, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
            for lbl in range(1, num_l):
                if stats[lbl, cv2.CC_STAT_AREA] >= (src_w * src_h * 0.001):
                    x, y, w, h, _ = stats[lbl]
                    gx0 = min(gx0, int(x))
                    gy0 = min(gy0, int(y))
                    gx1 = max(gx1, int(x + w))
                    gy1 = max(gy1, int(y + h))

    cap.release()
    del predictor, sam2_model
    torch.mps.empty_cache()

    # Safety margin: horizontal +50px, vertical +35px, clamped by clean regions
    sx0 = max(int(src_w * 0.15), gx0 - 54)
    sy0 = max(top_y_max, gy0 - 38)
    sx1 = min(int(src_w * 0.85), gx1 + 32)
    sy1 = min(bot_y_min, gy1 + 6)

    scale_x = TARGET_W / float(src_w)
    scale_y = TARGET_H / float(src_h)

    env = {
        "src_resolution": [src_w, src_h],
        "target_resolution": [TARGET_W, TARGET_H],
        "num_keyframes_scanned": len(sample_sec),
        "raw_union_src": [int(gx0), int(gy0), int(gx1), int(gy1)],
        "stage_src": [int(sx0), int(sy0), int(sx1), int(sy1)],
        "stage_target": [
            int(round(sx0 * scale_x)),
            int(round(sy0 * scale_y)),
            int(round(sx1 * scale_x)),
            int(round(sy1 * scale_y)),
        ],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    print(
        f"[MotionEnvelope] Global Union src={env['raw_union_src']} -> Safe Stage src={env['stage_src']} "
        f"-> target={env['stage_target']}",
        flush=True,
    )
    return env


# ==============================================================================
# 2. Dynamic In-Stage Overlay Cleaning & Seamless Peripheral Extension
# ==============================================================================
def remove_countdown_digit_dynamic(frame_bgr: np.ndarray, bg_ref: np.ndarray, region_cfg: dict) -> np.ndarray:
    """
    Dynamically remove translucent/white center countdown digits within region_cfg.
    """
    out = frame_bgr.copy()
    y0 = region_cfg.get("y0", 274)
    y1 = region_cfg.get("y1", 468)
    x0 = region_cfg.get("x0", 450)
    x1 = region_cfg.get("x1", 785)

    roi = out[y0:y1, x0:x1]
    bg_roi = bg_ref[y0:y1, x0:x1]

    b, g, r = roi[:, :, 0].astype(int), roi[:, :, 1].astype(int), roi[:, :, 2].astype(int)
    min_c = np.minimum(np.minimum(b, g), r)
    max_c = np.maximum(np.maximum(b, g), r)
    diff_l1 = np.sum(np.abs(roi.astype(int) - bg_roi.astype(int)), axis=2)

    # Achromatic white digits (never triggers on warm skin where R - B > 35)
    white_digit = ((min_c > 192) & ((max_c - min_c) < 26) & (diff_l1 > 60)).astype(np.uint8) * 255
    if not np.any(white_digit):
        return out

    white_dil = cv2.dilate(white_digit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    white_halo = cv2.dilate(white_digit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    raw_coach = ((diff_l1 > 65) & (white_halo == 0)).astype(np.uint8) * 255
    # The top 15% of digit box is background curtain/wall
    top_cut = int((y1 - y0) * 0.18)
    raw_coach[:top_cut, :] = 0
    raw_coach = cv2.morphologyEx(raw_coach, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(raw_coach, connectivity=8)
    clean_coach = np.zeros_like(raw_coach)
    for lbl in range(1, num_l):
        if stats[lbl, cv2.CC_STAT_AREA] >= 120:
            clean_coach[labels == lbl] = 255

    clean_coach = cv2.dilate(clean_coach, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 15)))

    h_roi, w_roi = clean_coach.shape
    for cx in range(w_roi):
        ys = np.where(clean_coach[:, cx] > 0)[0]
        if len(ys) > 0:
            clean_coach[ys[0]:, cx] = 255

    coach_mask = clean_coach > 0

    bg_rep = (white_dil > 0) & (~coach_mask)
    roi[bg_rep] = bg_roi[bg_rep]

    inp_m = ((white_dil > 0) & coach_mask).astype(np.uint8) * 255
    if np.any(inp_m):
        roi[:] = cv2.inpaint(roi, inp_m, 5, cv2.INPAINT_TELEA)

    out[y0:y1, x0:x1] = roi
    return out


def prepare_stage_assets(
    video_path: Path,
    timeline_data: dict,
    envelope: dict,
    work_dir: Path,
    bg_arg: str = None
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Construct:
    1) bg_clean_ref: Static clean background plate for stage inpainting & edge sampling.
    2) clean_ready_pool: De-overlayed solid coach frames for intro (if intro exists).
    3) bg_peripheral_1080: 1080P same-color seamless extended studio background.
    4) stage_alpha_1080: Feathered stage blend mask.
    """
    cap = cv2.VideoCapture(str(video_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    clean_cfg = timeline_data.get("clean_regions", {})
    top_bar_y = clean_cfg.get("top_bar", {}).get("y_max", int(src_h * 0.38))
    bot_bar_y = clean_cfg.get("bottom_bar", {}).get("y_min", int(src_h * 0.825))
    countdown_cfg = clean_cfg.get("center_countdown", {
        "x0": int(src_w * 0.35), "y0": top_bar_y, "x1": int(src_w * 0.62), "y1": int(src_h * 0.65)
    })
    open_cfg = clean_cfg.get("opening_card", None)

    # 1. Build clean background reference plate (bg_clean_ref)
    # Sample 10 frames across the video and take temporal median on static court areas
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(int(total_frames * 0.05), int(total_frames * 0.95), 15).astype(int)
    frames_sample = []
    for f_i in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_i)
        r, f = cap.read()
        if r and f is not None:
            frames_sample.append(f)
    cap.release()

    if len(frames_sample) == 0:
        raise RuntimeError("Failed to read sample frames from input video.")

    # High-quality reference frame (e.g. prone stretch or rest where athlete is low)
    bg_clean_ref = np.median(frames_sample, axis=0).astype(np.uint8)

    # Clean top menu & bottom subtitle on bg_clean_ref by sampling neighboring court rows
    # Top wall: replicate row top_bar_y + 10 upward
    top_wall_row = bg_clean_ref[top_bar_y + 8:top_bar_y + 20, :].mean(axis=0, keepdims=True).astype(np.uint8)
    for y in range(top_bar_y):
        bg_clean_ref[y, :] = top_wall_row[0]

    # Bottom floor: replicate row bot_bar_y - 10 downward
    bot_floor_row = bg_clean_ref[bot_bar_y - 15:bot_bar_y - 5, :].mean(axis=0, keepdims=True).astype(np.uint8)
    for y in range(bot_bar_y, src_h):
        bg_clean_ref[y, :] = bot_floor_row[0]

    # 2. Build clean_ready_pool for intro (if opening_card is specified)
    clean_ready_pool = []
    if open_cfg:
        intro_start = float(open_cfg.get("start_sec", 0.0))
        intro_end = float(open_cfg.get("end_sec", 3.0))
        # Grab frames right after intro_end (when title card is gone)
        cap = cv2.VideoCapture(str(video_path))
        start_ready_f = int(round((intro_end + 0.05) * fps))
        for offset_f in range(8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_ready_f + offset_f)
            r, fr = cap.read()
            if r and fr is not None:
                c_fr = fr.copy()
                c_fr[:top_bar_y, :] = bg_clean_ref[:top_bar_y, :]
                c_fr[bot_bar_y:, :] = bg_clean_ref[bot_bar_y:, :]
                c_fr = remove_countdown_digit_dynamic(c_fr, bg_clean_ref, countdown_cfg)
                clean_ready_pool.append(c_fr)
        cap.release()

    # 3. Build Seamless Same-Color Peripheral Background (1920x1080)
    sx0_src, sy0_src, sx1_src, sy1_src = envelope["stage_src"]
    tx0, ty0, tx1, ty1 = envelope["stage_target"]

    base_src = bg_clean_ref.copy()
    # Sample top edge color profile of the stage (wall/curtain)
    top_strip = base_src[sy0_src:sy0_src + 20, :].astype(np.float32)
    top_profile = cv2.GaussianBlur(np.median(top_strip, axis=0, keepdims=True), (151, 1), 40.0)
    for y in range(0, sy0_src):
        t_w = np.clip((sy0_src - y) / float(max(1, sy0_src)), 0.0, 1.0)
        base_src[y, :] = np.clip(top_profile[0] * (1.0 - 0.12 * t_w), 0, 255).astype(np.uint8)

    # Sample bottom edge color profile of the stage (floor)
    bot_strip = base_src[sy1_src - 20:sy1_src, :].astype(np.float32)
    bot_profile = cv2.GaussianBlur(np.median(bot_strip, axis=0, keepdims=True), (151, 1), 40.0)
    for y in range(sy1_src, src_h):
        base_src[y, :] = bot_profile[0].astype(np.uint8)

    # Smooth out sides (left and right) outside stage
    for x in range(0, sx0_src):
        t_x = np.clip((sx0_src - x) / float(max(1, sx0_src)), 0.0, 1.0)
        side_col = cv2.GaussianBlur(base_src[:, sx0_src:sx0_src + 15].astype(np.float32), (1, 31), 8.0).mean(axis=1)
        base_src[:, x] = (base_src[:, x].astype(np.float32) * (1.0 - 0.65 * t_x) + side_col * (0.65 * t_x)).astype(np.uint8)

    for x in range(sx1_src, src_w):
        t_x = np.clip((x - sx1_src) / float(max(1, src_w - sx1_src)), 0.0, 1.0)
        side_col = cv2.GaussianBlur(base_src[:, sx1_src - 15:sx1_src].astype(np.float32), (1, 31), 8.0).mean(axis=1)
        base_src[:, x] = (base_src[:, x].astype(np.float32) * (1.0 - 0.65 * t_x) + side_col * (0.65 * t_x)).astype(np.uint8)

    # Upscale base_src to 1080P with color enhancement matching foreground
    bg_1080 = cv2.resize(base_src, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(bg_1080, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.03 + 2.0, 0, 255)
    bg_1080 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if bg_arg and os.path.exists(bg_arg):
        custom_bg = cv2.imread(bg_arg)
        if custom_bg is not None:
            bg_1080 = cv2.resize(custom_bg, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    # Multi-scale matte blur and radial vignette outside Core Stage
    bg_blur_near = cv2.GaussianBlur(bg_1080, (45, 45), 14.0).astype(np.float32)
    bg_blur_far = cv2.GaussianBlur(bg_1080, (121, 121), 42.0).astype(np.float32)

    y_grid, x_grid = np.mgrid[0:TARGET_H, 0:TARGET_W].astype(np.float32)
    dx = np.maximum(0.0, np.maximum(tx0 - x_grid, x_grid - tx1))
    dy = np.maximum(0.0, np.maximum(ty0 - y_grid, y_grid - ty1))
    dist_out = np.sqrt(dx ** 2 + dy ** 2)

    t_far = np.clip(dist_out / 180.0, 0.0, 1.0)[:, :, None]
    bg_matte = bg_blur_near * (1.0 - t_far) + bg_blur_far * t_far
    vig = 1.0 - 0.22 * np.clip(dist_out / 520.0, 0.0, 1.0) ** 1.35
    bg_peripheral_1080 = np.clip(bg_matte * vig[:, :, None], 0, 255).astype(np.uint8)

    # 4. Feathered Core Stage alpha mask
    s_pil = Image.new("L", (TARGET_W, TARGET_H), 0)
    s_draw = ImageDraw.Draw(s_pil)
    s_draw.rounded_rectangle([tx0 - 12, ty0 - 12, tx1 + 12, ty1 + 12], radius=48, fill=255)
    stage_u8 = np.array(s_pil)
    stage_alpha_1080 = (cv2.GaussianBlur(stage_u8, (75, 75), 22.0).astype(np.float32) / 255.0)[:, :, None]

    return bg_clean_ref, clean_ready_pool, bg_peripheral_1080, stage_alpha_1080


def clean_stage_frame_dynamic(
    frame_bgr: np.ndarray,
    sec_float: float,
    global_f_idx: int,
    fps: float,
    timeline_data: dict,
    bg_clean_ref: np.ndarray,
    clean_ready_pool: list[np.ndarray],
) -> np.ndarray:
    """
    Clean frame overlays dynamically based on timeline clean_regions and workout intervals.
    """
    clean_cfg = timeline_data.get("clean_regions", {})
    open_cfg = clean_cfg.get("opening_card", None)

    # 1. Opening card replacement
    if open_cfg and len(clean_ready_pool) > 0:
        start_s = float(open_cfg.get("start_sec", 0.0))
        end_s = float(open_cfg.get("end_sec", 3.0))
        if start_s <= sec_float <= end_s:
            pool_n = len(clean_ready_pool) - 1
            dur_f = max(1.0, (end_s - start_s) * fps)
            phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * (sec_float - start_s) * fps / dur_f))
            p_idx = min(pool_n, int(round(phase * pool_n)))
            return clean_ready_pool[p_idx].copy()

    out = frame_bgr.copy()
    src_h, src_w = out.shape[:2]
    top_bar_y = clean_cfg.get("top_bar", {}).get("y_max", int(src_h * 0.38))
    bot_bar_y = clean_cfg.get("bottom_bar", {}).get("y_min", int(src_h * 0.825))

    out[:top_bar_y, :] = bg_clean_ref[:top_bar_y, :]
    out[bot_bar_y:, :] = bg_clean_ref[bot_bar_y:, :]

    # 2. Check if current time is inside a rest period or intro countdown
    is_rest_or_intro = False
    intro = timeline_data.get("intro", {})
    if intro and intro.get("start_sec", 0) <= sec_float < intro.get("end_sec", 0):
        # Active intro countdown
        if sec_float >= float(intro.get("start_sec", 0)) + 2.95:
            is_rest_or_intro = True

    for b in timeline_data.get("blocks", []):
        r_start = float(b.get("rest_start_sec", 0))
        r_end = float(b.get("rest_end_sec", 0))
        if r_start <= sec_float < r_end:
            is_rest_or_intro = True
            break

    if is_rest_or_intro:
        countdown_cfg = clean_cfg.get("center_countdown", {
            "x0": int(src_w * 0.35), "y0": top_bar_y, "x1": int(src_w * 0.62), "y1": int(src_h * 0.65)
        })
        out = remove_countdown_digit_dynamic(out, bg_clean_ref, countdown_cfg)

    # 3. Outro text removal
    outro_cfg = clean_cfg.get("outro_text", None)
    if outro_cfg:
        o_start = float(outro_cfg.get("start_sec", 1e9))
        o_end = float(outro_cfg.get("end_sec", -1))
        if o_start <= sec_float < o_end:
            ox0, oy0 = outro_cfg.get("x0", 0), outro_cfg.get("y0", 0)
            ox1, oy1 = outro_cfg.get("x1", src_w), outro_cfg.get("y1", src_h)
            out[oy0:oy1, ox0:ox1] = bg_clean_ref[oy0:oy1, ox0:ox1]

    return out


def enhance_foreground_1080p(frame_src: np.ndarray) -> np.ndarray:
    """Upscale frame to 1080P with Lanczos4 and apply gentle sharpening and color pop."""
    up = cv2.resize(frame_src, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(up, (0, 0), 1.2)
    sharp = cv2.addWeighted(up, 1.22, blurred, -0.22, 0)
    hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.03 + 2.0, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ==============================================================================
# 3. Dynamic English UI Renderer (Zero Text Overlaps, N Exercises)
# ==============================================================================
class WorkoutUIRenderer:
    def __init__(self, timeline_data: dict):
        self.timeline = timeline_data
        self.blocks = timeline_data.get("blocks", [])
        self.total_blocks = len(self.blocks)
        self.total_dur = float(timeline_data.get("video_info", {}).get("total_duration_sec", 335.0))
        self.workout_title = timeline_data.get("video_info", {}).get("title_en", "WORKOUT")

        self.font_badge = ImageFont.truetype(FONT_PATH, 17, index=1)
        self.font_title = ImageFont.truetype(FONT_PATH, 42, index=1)
        self.font_sub = ImageFont.truetype(FONT_PATH, 23, index=0)
        self.font_timer = ImageFont.truetype(FONT_PATH, 72, index=1)
        self.font_timer_lbl = ImageFont.truetype(FONT_PATH, 16, index=1)
        self.font_next_lbl = ImageFont.truetype(FONT_PATH, 18, index=1)
        self.font_next_val_26 = ImageFont.truetype(FONT_PATH, 26, index=1)
        self.font_next_val_23 = ImageFont.truetype(FONT_PATH, 23, index=1)
        self.font_next_val_20 = ImageFont.truetype(FONT_PATH, 20, index=1)

    def get_state_at_time(self, t: float) -> dict:
        intro = self.timeline.get("intro", {})
        intro_dur = float(intro.get("duration_sec", 0.0)) if intro else 0.0
        if intro_dur > 0 and t < float(intro.get("end_sec", intro_dur)):
            i_end = float(intro.get("end_sec", intro_dur))
            rem = max(0, int(math.ceil(i_end - t)))
            prog = np.clip(t / max(1.0, i_end), 0.0, 1.0)
            first_name = self.blocks[0]["name_en"] if self.blocks else "First Movement"
            return {
                "phase": "INTRO",
                "block_idx": 0,
                "total_blocks": self.total_blocks,
                "title": "GET READY: WORKOUT",
                "subtitle": self.workout_title,
                "next_title": f"1. {first_name}",
                "rem_sec": rem,
                "phase_progress": prog,
                "overall_progress": t / max(1.0, self.total_dur),
                "is_final_321": (0 < (i_end - t) <= 3.1),
            }

        for idx, b in enumerate(self.blocks):
            b_num = b.get("id", idx + 1)
            start_s = float(b["start_sec"])
            end_s = float(b["end_sec"])
            dur_s = float(b.get("duration_sec", end_s - start_s))
            r_start = float(b.get("rest_start_sec", end_s))
            r_end = float(b.get("rest_end_sec", r_start))
            r_dur = float(b.get("rest_duration_sec", r_end - r_start))

            # Active Exercise
            if start_s <= t < end_s:
                elapsed = t - start_s
                rem = max(0, int(math.ceil(dur_s - elapsed)))
                prog = np.clip(elapsed / max(1.0, dur_s), 0.0, 1.0)
                return {
                    "phase": "WORK",
                    "block_idx": b_num,
                    "total_blocks": self.total_blocks,
                    "title": b["name_en"],
                    "subtitle": b.get("coach_cue_en", "Focus on core control & steady breathing."),
                    "next_title": b.get("next_name_en", "Next Movement"),
                    "rem_sec": rem,
                    "phase_progress": prog,
                    "overall_progress": t / max(1.0, self.total_dur),
                    "is_final_321": (0 < (end_s - t) <= 3.1),
                }

            # Rest Period
            if r_start <= t < r_end:
                is_last_block = (idx == len(self.blocks) - 1)
                elapsed = t - r_start
                rem = max(0, int(math.ceil(r_dur - elapsed)))
                prog = np.clip(elapsed / max(1.0, r_dur), 0.0, 1.0)
                if is_last_block:
                    return {
                        "phase": "OUTRO",
                        "block_idx": b_num,
                        "total_blocks": self.total_blocks,
                        "title": "WORKOUT COMPLETE!",
                        "subtitle": "Great job! Your core is primed and energized.",
                        "next_title": "Recovery & Cool Down",
                        "rem_sec": rem,
                        "phase_progress": 1.0,
                        "overall_progress": min(1.0, t / max(1.0, self.total_dur)),
                        "is_final_321": False,
                    }
                else:
                    next_idx = b_num + 1
                    return {
                        "phase": "REST",
                        "block_idx": b_num,
                        "total_blocks": self.total_blocks,
                        "title": "ACTIVE RECOVERY & REST",
                        "subtitle": "Breathe deeply and prepare for the next movement",
                        "next_title": f"{next_idx}. {b.get('next_name_en', '')}",
                        "rem_sec": rem,
                        "phase_progress": prog,
                        "overall_progress": t / max(1.0, self.total_dur),
                        "is_final_321": (0 < (r_end - t) <= 3.1),
                    }

        # Outro fallback
        return {
            "phase": "OUTRO",
            "block_idx": self.total_blocks,
            "total_blocks": self.total_blocks,
            "title": "WORKOUT COMPLETE!",
            "subtitle": f"{self.workout_title} Finished",
            "next_title": "Done",
            "rem_sec": 0,
            "phase_progress": 1.0,
            "overall_progress": 1.0,
            "is_final_321": False,
        }

    def _draw_wrapped_next_title(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, text: str, max_w: int, fill_col: tuple
    ) -> int:
        for fnt in (self.font_next_val_26, self.font_next_val_23, self.font_next_val_20):
            bbox = draw.textbbox((0, 0), text, font=fnt)
            if (bbox[2] - bbox[0]) <= max_w:
                draw.text((x, y), text, font=fnt, fill=fill_col)
                return y + (bbox[3] - bbox[1]) + 6

        fnt = self.font_next_val_23
        words = text.split(" ")
        line1, line2 = text, ""
        for split_i in range(len(words) - 1, 0, -1):
            cand1 = " ".join(words[:split_i])
            cand2 = " ".join(words[split_i:])
            w1 = draw.textbbox((0, 0), cand1, font=fnt)[2]
            w2 = draw.textbbox((0, 0), cand2, font=fnt)[2]
            if w1 <= max_w and w2 <= max_w:
                line1, line2 = cand1, cand2
                break
        else:
            fnt = self.font_next_val_20
            for split_i in range(len(words) - 1, 0, -1):
                cand1 = " ".join(words[:split_i])
                cand2 = " ".join(words[split_i:])
                if draw.textbbox((0, 0), cand1, font=fnt)[2] <= max_w:
                    line1, line2 = cand1, cand2
                    break

        draw.text((x, y - 4), line1, font=fnt, fill=fill_col)
        if line2:
            draw.text((x, y + 22), line2, font=fnt, fill=fill_col)
            return y + 50
        return y + 26

    def render_ui(self, frame_bgr: np.ndarray, t: float) -> np.ndarray:
        state = self.get_state_at_time(t)
        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        phase = state["phase"]
        n_blocks = max(1, state["total_blocks"])
        if phase == "WORK":
            accent = (0, 230, 168, 255)
            badge_txt = f"EXERCISE {state['block_idx']} OF {n_blocks}"
        elif phase == "REST":
            accent = (255, 165, 38, 255)
            badge_txt = f"REST  •  COMPLETED {state['block_idx']}/{n_blocks}"
        elif phase == "INTRO":
            accent = (56, 189, 248, 255)
            badge_txt = f"WORKOUT  •  {n_blocks} EXERCISES"
        else:
            accent = (168, 85, 247, 255)
            badge_txt = f"SESSION COMPLETE  •  {n_blocks}/{n_blocks} FINISHED"

        # 1. Top-Left Card: x=[56, 1180], y=[36, 206]
        draw.rounded_rectangle(
            [56, 36, 1180, 206], radius=20, fill=(16, 20, 28, 210), outline=(255, 255, 255, 38), width=2
        )
        draw.rounded_rectangle([56, 36, 70, 206], radius=6, fill=accent)

        tb = draw.textbbox((0, 0), badge_txt, font=self.font_badge)
        pill_w = (tb[2] - tb[0]) + 28
        pill_x1 = min(530, 92 + pill_w)
        draw.rounded_rectangle([92, 52, pill_x1, 82], radius=8, fill=(accent[0], accent[1], accent[2], 38))
        draw.text((106, 57), badge_txt, font=self.font_badge, fill=accent)

        # Dynamic N-block segmented progress bar (fitting dynamically in x: 575..1147)
        avail_bar_w = 1147 - 575
        seg_gap = 6 if n_blocks > 10 else 8
        total_gaps = (n_blocks - 1) * seg_gap
        seg_w = max(12, int((avail_bar_w - total_gaps) / n_blocks))
        bar_start_x = 1147 - (n_blocks * seg_w + total_gaps)
        bar_start_x = max(550, bar_start_x)

        for i in range(n_blocks):
            sx0 = bar_start_x + i * (seg_w + seg_gap)
            sx1 = sx0 + seg_w
            if (i + 1) < state["block_idx"] or phase == "OUTRO":
                s_col = (0, 230, 168, 215)
            elif (i + 1) == state["block_idx"] and phase == "WORK":
                s_col = accent
            elif (i + 1) == state["block_idx"] and phase == "REST":
                s_col = (0, 230, 168, 215)
            else:
                s_col = (255, 255, 255, 45)
            draw.rounded_rectangle([sx0, 62, sx1, 72], radius=4, fill=s_col)

        draw.text((92, 94), state["title"], font=self.font_title, fill=(255, 255, 255, 255))
        draw.text((92, 154), state["subtitle"], font=self.font_sub, fill=(195, 205, 220, 235))

        # 2. Top-Right Card: x=[1216, 1864], y=[36, 206]
        draw.rounded_rectangle(
            [1216, 36, 1864, 206], radius=20, fill=(16, 20, 28, 210), outline=(255, 255, 255, 38), width=2
        )
        draw.line([(1604, 54), (1604, 158)], fill=(255, 255, 255, 28), width=2)

        rem = state["rem_sec"]
        timer_str = f"00:{rem:02d}"
        timer_col = (255, 95, 95, 255) if state["is_final_321"] else (255, 255, 255, 255)
        draw.text((1630, 52), "TIME REMAINING", font=self.font_timer_lbl, fill=(160, 175, 195, 220))
        draw.text((1625, 74), timer_str, font=self.font_timer, fill=timer_col)

        MAX_LEFT_W = 342
        if phase in ("REST", "INTRO"):
            draw.text((1244, 50), "UP NEXT", font=self.font_next_lbl, fill=accent)
            bot_y = self._draw_wrapped_next_title(
                draw, 1244, 78, state["next_title"], MAX_LEFT_W, (255, 255, 255, 250)
            )
            if state["is_final_321"] and rem > 0:
                pill_y0 = max(130, bot_y + 2)
                draw.rounded_rectangle([1244, pill_y0, 1445, pill_y0 + 28], radius=7, fill=(255, 80, 80, 215))
                draw.text((1258, pill_y0 + 5), f"STARTING IN {rem}...", font=self.font_badge, fill=(255, 255, 255, 255))
        elif phase == "WORK":
            draw.text((1244, 50), "NEXT MOVEMENT", font=self.font_timer_lbl, fill=(150, 165, 185, 200))
            bot_y = self._draw_wrapped_next_title(
                draw, 1244, 76, state["next_title"], MAX_LEFT_W, (225, 232, 240, 240)
            )
            draw.text((1244, max(126, bot_y + 4)), "TARGET: CORE STABILITY", font=self.font_timer_lbl, fill=accent)
        else:
            draw.text((1244, 65), "GREAT WORKOUT!", font=self.font_next_val_26, fill=accent)
            draw.text((1244, 110), "100% Core Completed", font=self.font_sub, fill=(225, 232, 240, 235))

        bar_x0, bar_x1 = 1244, 1834
        bar_y0, bar_y1 = 174, 188
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=7, fill=(255, 255, 255, 38))
        fill_w = int(
            (bar_x1 - bar_x0)
            * np.clip(1.0 - state["phase_progress"] if phase in ("WORK", "REST", "INTRO") else 1.0, 0.02, 1.0)
        )
        draw.rounded_rectangle(
            [bar_x0, bar_y0, bar_x0 + fill_w, bar_y1], radius=7, fill=timer_col if state["is_final_321"] else accent
        )

        out_pil = Image.alpha_composite(img_pil, overlay).convert("RGB")
        return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)


# ==============================================================================
# 4. Dynamic Audio System (BGM, Beeps, TTS Voice & Auto Ducking)
# ==============================================================================
def generate_120bpm_bgm(duration_sec: float, work_dir: Path, bgm_arg=None) -> np.ndarray:
    bgm_dir = work_dir / "assets" / "bgm"
    bgm_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * SAMPLE_RATE)

    custom_path = None
    if bgm_arg and os.path.exists(bgm_arg):
        custom_path = bgm_arg
    else:
        existing = sorted(glob.glob(str(bgm_dir / "*.wav")) + glob.glob(str(bgm_dir / "*.mp3")))
        if existing and "placeholder_120bpm.wav" not in existing[0]:
            custom_path = existing[0]

    if custom_path:
        tmp_wav = work_dir / "tmp" / "custom_bgm_44k.wav"
        tmp_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(custom_path),
                "-ac", "2", "-ar", str(SAMPLE_RATE), "-t", str(duration_sec),
                str(tmp_wav),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tmp_wav.exists():
            with wave.open(str(tmp_wav), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                audio = audio.reshape(-1, 2)
                if len(audio) < n_samples:
                    reps = int(math.ceil(n_samples / len(audio)))
                    audio = np.tile(audio, (reps, 1))[:n_samples]
                else:
                    audio = audio[:n_samples]
                peak = np.max(np.abs(audio)) + 1e-6
                return (audio / peak) * 0.18

    print("[Audio] Synthesizing 120BPM rhythmic workout BGM (base volume 18%)...", flush=True)
    t = np.arange(n_samples, dtype=np.float32) / SAMPLE_RATE
    beat_dur = 0.5

    beat_phase = np.mod(t, beat_dur)
    kick_freq = 52.0 + 95.0 * np.exp(-beat_phase * 38.0)
    kick_env = np.exp(-beat_phase * 11.5) * (beat_phase < 0.35)
    kick = np.sin(2.0 * np.pi * kick_freq * beat_phase) * kick_env * 0.65

    rng = np.random.default_rng(123)
    noise = rng.uniform(-1.0, 1.0, n_samples).astype(np.float32)
    hat_phase = np.mod(t + 0.25, beat_dur)
    hat_env = np.exp(-hat_phase * 45.0) * (hat_phase < 0.12)
    hihat = noise * hat_env * 0.22

    bar_idx = np.floor(np.mod(t, 8.0) / 2.0).astype(int)
    bass_freqs = np.choose(bar_idx, [55.0, 43.65, 65.41, 49.0])
    sub_phase = np.mod(t, 0.25)
    bass_env = (1.0 - np.exp(-beat_phase * 16.0)) * np.exp(-sub_phase * 7.5)
    bass = (np.sin(2.0 * np.pi * bass_freqs * t) + 0.35 * np.sin(4.0 * np.pi * bass_freqs * t)) * bass_env * 0.42

    arp_idx = np.floor(np.mod(t, 1.0) / 0.25).astype(int)
    arp_mult = np.choose(arp_idx, [4.0, 5.0, 6.0, 5.0])
    arp_freq = bass_freqs * arp_mult
    arp_env = np.exp(-sub_phase * 14.0)
    arp_l = np.sin(2.0 * np.pi * arp_freq * t) * arp_env * 0.16
    arp_r = np.sin(2.0 * np.pi * (arp_freq * 1.003) * t + 0.4) * arp_env * 0.16

    left = kick + hihat + bass + arp_l
    right = kick + hihat + bass + arp_r
    stereo = np.stack([left, right], axis=1)

    peak = np.max(np.abs(stereo)) + 1e-6
    bgm_18 = (stereo / peak) * 0.18

    out_wav = bgm_dir / "placeholder_120bpm.wav"
    if not out_wav.exists():
        pcm = np.clip(bgm_18 * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

    return bgm_18


def make_beep(freq_hz: float, dur_sec: float, amp: float = 0.52) -> np.ndarray:
    n = int(dur_sec * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    wave_1d = (np.sin(2.0 * np.pi * freq_hz * t) + 0.18 * np.sin(4.0 * np.pi * freq_hz * t)) * amp
    fade_n = min(int(0.015 * SAMPLE_RATE), n // 4)
    if fade_n > 0:
        wave_1d[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        wave_1d[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    return np.stack([wave_1d, wave_1d], axis=1)


def build_beep_track(timeline_data: dict, total_dur: float) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(total_dur * SAMPLE_RATE)
    beep_track = np.zeros((n_samples, 2), dtype=np.float32)
    beep_env = np.zeros(n_samples, dtype=np.float32)

    def place_tone(time_sec: float, tone: np.ndarray):
        idx0 = int(round(time_sec * SAMPLE_RATE))
        idx1 = min(n_samples, idx0 + len(tone))
        if 0 <= idx0 < n_samples and idx1 > idx0:
            seg_len = idx1 - idx0
            beep_track[idx0:idx1] += tone[:seg_len]
            beep_env[idx0:idx1] = 1.0

    beep_rest = make_beep(440.0, 0.40, amp=0.50)
    beep_cnt = make_beep(880.0, 0.15, amp=0.50)
    beep_go = make_beep(1046.0, 0.35, amp=0.55)

    prev_end_s = 0.0
    blocks = timeline_data.get("blocks", [])
    for idx, b in enumerate(blocks):
        start_s = float(b["start_sec"])
        end_s = float(b["end_sec"])
        if idx == 0 or (start_s - prev_end_s) >= 3.5:
            for offset in (3.0, 2.0, 1.0):
                place_tone(start_s - offset, beep_cnt)
        place_tone(start_s, beep_go)
        place_tone(end_s, beep_rest)
        prev_end_s = end_s

    return beep_track, beep_env


async def synthesize_tts_clips(timeline_data: dict, tts_cache_dir: Path) -> list[dict]:
    tts_cache_dir.mkdir(parents=True, exist_ok=True)
    events = []

    intro = timeline_data.get("intro", {})
    intro_dur = float(intro.get("duration_sec", 0.0)) if intro else 0.0
    first_name = timeline_data.get("blocks", [{}])[0].get("name_en", "First Exercise")
    workout_title = timeline_data.get("video_info", {}).get("title_en", "Workout")

    if intro_dur > 0:
        events.append(
            {
                "key": "intro_welcome",
                "text": f"Welcome to the {workout_title}! Get on the mat. First up: {first_name}.",
                "rate": "+8%",
                "align": "start",
                "target_sec": float(intro.get("start_sec", 0.0)) + 0.20,
            }
        )

    events.append(
        {
            "key": "countdown_321_go",
            "text": "Three, two, one, Go!",
            "rate": "+5%",
            "align": "countdown_go",
            "target_sec": 13.0,
        }
    )

    blocks = timeline_data.get("blocks", [])
    for idx, b in enumerate(blocks):
        bid = b.get("id", idx + 1)
        start_s = float(b["start_sec"])
        end_s = float(b["end_sec"])
        is_last = (idx == len(blocks) - 1)
        r_dur = float(b.get("rest_duration_sec", float(b.get("rest_end_sec", end_s)) - end_s))

        # Mid-exercise cue
        cue = b.get("coach_cue_en", "Stay strong and maintain smooth form.")
        events.append(
            {
                "key": f"block_{bid:02d}_mid",
                "text": cue,
                "rate": "+8%",
                "align": "start",
                "target_sec": start_s + 6.2,
            }
        )

        # Rest announcement
        if not is_last and r_dur >= 5.0:
            next_name = b.get("next_name_en", "")
            events.append(
                {
                    "key": f"block_{bid:02d}_rest",
                    "text": f"Rest and recover. Next up: {next_name}." if next_name else "Rest and recover.",
                    "rate": "+8%",
                    "align": "start",
                    "target_sec": end_s + 0.55,
                }
            )
        elif not is_last and r_dur < 5.0:
            next_name = b.get("next_name_en", "")
            events.append(
                {
                    "key": f"block_{bid:02d}_quick",
                    "text": f"Next movement: {next_name}!",
                    "rate": "+10%",
                    "align": "start",
                    "target_sec": end_s + 0.1,
                }
            )

    outro = timeline_data.get("outro", {})
    if outro:
        o_start = float(outro.get("start_sec", float(blocks[-1]["end_sec"]) if blocks else 317.0))
        events.append(
            {
                "key": "outro_complete",
                "text": "Workout complete! Awesome effort today. Your core is stronger, more stable, and ready to dominate the court. See you next time!",
                "rate": "+6%",
                "align": "start",
                "target_sec": o_start + 0.6,
            }
        )

    for ev in events:
        mp3_path = tts_cache_dir / f"{ev['key']}.mp3"
        wav_path = tts_cache_dir / f"{ev['key']}.wav"
        if not wav_path.exists() or wav_path.stat().st_size < 1000:
            success = False
            for attempt in range(8):
                try:
                    rate_val = ev["rate"] if attempt < 4 else "+0%"
                    comm = edge_tts.Communicate(ev["text"], TTS_VOICE, rate=rate_val)
                    await comm.save(str(mp3_path))
                    if mp3_path.exists() and mp3_path.stat().st_size > 200:
                        success = True
                        break
                except Exception as e:
                    print(f"[TTS Retry {attempt+1}] {ev['key']}: {e}", flush=True)
                    await asyncio.sleep(1.2 * (attempt + 1))
            if not success:
                subprocess.run(
                    [
                        "/Users/guo/.venv/bin/edge-tts",
                        "--voice", TTS_VOICE,
                        "--text", ev["text"],
                        "--write-media", str(mp3_path),
                    ],
                    check=True,
                )
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp3_path), "-ac", "2", "-ar", str(SAMPLE_RATE), str(wav_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            await asyncio.sleep(0.4)
        ev["wav_path"] = wav_path

    return events


def load_wav_float32(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        arr = arr.reshape(-1, 2)
        env = np.max(np.abs(arr), axis=1)
        nz = np.where(env > 0.006)[0]
        if len(nz) > 0:
            i0 = max(0, nz[0] - int(0.02 * SAMPLE_RATE))
            i1 = min(len(arr), nz[-1] + int(0.04 * SAMPLE_RATE))
            arr = arr[i0:i1]
        peak = np.max(np.abs(arr)) + 1e-6
        return (arr / peak) * 0.72


def build_full_master_audio(timeline_data: dict, total_dur: float, work_dir: Path, bgm_arg=None) -> Path:
    out_wav = work_dir / "tmp" / "master_audio_full.wav"
    if bgm_arg is None and out_wav.exists() and out_wav.stat().st_size > 50_000_000:
        print(f"[Audio] Reusing cached master audio track at {out_wav}", flush=True)
        return out_wav

    tts_cache_dir = work_dir / "tmp" / "tts_cache"
    events = asyncio.run(synthesize_tts_clips(timeline_data, tts_cache_dir))

    n_samples = int(total_dur * SAMPLE_RATE)
    bgm_track = generate_120bpm_bgm(total_dur, work_dir, bgm_arg=bgm_arg)
    beep_track, beep_env = build_beep_track(timeline_data, total_dur)
    voice_track = np.zeros((n_samples, 2), dtype=np.float32)
    voice_env = np.zeros(n_samples, dtype=np.float32)

    def place_voice(start_sec: float, clip: np.ndarray):
        idx0 = max(0, int(round(start_sec * SAMPLE_RATE)))
        idx1 = min(n_samples, idx0 + len(clip))
        if idx1 > idx0:
            seg_len = idx1 - idx0
            voice_track[idx0:idx1] += clip[:seg_len]
            voice_env[idx0:idx1] = 1.0

    countdown_clip = None
    for ev in events:
        clip = load_wav_float32(ev["wav_path"])
        if ev["key"] == "countdown_321_go":
            countdown_clip = clip
        else:
            place_voice(ev["target_sec"], clip)

    if countdown_clip is not None:
        clip_dur = len(countdown_clip) / SAMPLE_RATE
        blocks = timeline_data.get("blocks", [])
        for idx, b in enumerate(blocks):
            # Only play 3-2-1 countdown if preparation window >= 3.5s
            start_s = float(b["start_sec"])
            prev_end = float(blocks[idx - 1]["end_sec"]) if idx > 0 else 0.0
            if idx == 0 or (start_s - prev_end) >= 3.5:
                place_voice(start_s - max(3.0, clip_dur - 0.30), countdown_clip)

    duck_trigger = ((beep_env > 0) | (voice_env > 0)).astype(np.float32)
    step = 100
    trig_ds = duck_trigger[::step]
    trig_dil = cv2.dilate(trig_ds.reshape(-1, 1), np.ones((int(0.35 * SAMPLE_RATE / step), 1), np.uint8)).flatten()
    win_len = int(0.28 * SAMPLE_RATE / step) | 1
    trig_smooth_ds = cv2.GaussianBlur(trig_dil.reshape(-1, 1).astype(np.float32), (1, win_len), 0).flatten()
    trig_smooth = np.interp(np.arange(n_samples), np.arange(len(trig_smooth_ds)) * step, trig_smooth_ds)

    duck_gain = 1.0 - (1.0 - 0.3981) * np.clip(trig_smooth, 0.0, 1.0)
    bgm_ducked = bgm_track * duck_gain[:, np.newaxis]

    mix = bgm_ducked + beep_track * 0.48 + voice_track * 0.82

    peak = np.max(np.abs(mix)) + 1e-6
    target_ceiling = 0.82
    if peak > target_ceiling:
        mix = np.tanh(mix / target_ceiling) * target_ceiling
    else:
        mix = (mix / peak) * min(peak, target_ceiling)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(mix * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    print(
        f"[Audio] Master audio mixed and saved to {out_wav} (peak={20*math.log10(np.max(np.abs(mix))+1e-9):.2f} dBFS)",
        flush=True,
    )
    return out_wav


# ==============================================================================
# 5. Core Video Rendering Loop
# ==============================================================================
def render_video_segment(
    video_path: Path,
    timeline_data: dict,
    bg_clean_ref: np.ndarray,
    clean_ready_pool: list[np.ndarray],
    bg_peripheral_1080: np.ndarray,
    stage_alpha_1080: np.ndarray,
    master_wav: Path,
    out_mp4: Path,
    start_sec: float,
    end_sec: float,
    work_dir: Path,
):
    t0 = time.time()
    ui_renderer = WorkoutUIRenderer(timeline_data)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    start_f = int(round(start_sec * fps))
    end_f = int(round(end_sec * fps))
    total_f = end_f - start_f
    dur_sec = end_sec - start_sec

    seg_wav = work_dir / "tmp" / f"seg_audio_{int(start_sec)}_{int(end_sec)}.wav"
    seg_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-t", f"{dur_sec:.3f}",
            "-i", str(master_wav), "-ac", "2", "-ar", str(SAMPLE_RATE),
            str(seg_wav),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{TARGET_W}x{TARGET_H}",
        "-pix_fmt", "bgr24",
        "-r", str(int(round(fps))),
        "-i", "-",
        "-i", str(seg_wav),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out_mp4),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    bg_float = bg_peripheral_1080.astype(np.float32)
    total_dur = float(timeline_data.get("video_info", {}).get("total_duration_sec", 335.0))
    outro_fade_start = total_dur - 8.2

    for local_f in range(total_f):
        ret, frame_src = cap.read()
        if not ret or frame_src is None:
            frame_src = np.zeros_like(bg_clean_ref)

        global_f_idx = start_f + local_f
        sec_fl = global_f_idx / fps

        if sec_fl >= outro_fade_start:
            comp = (bg_float * 0.78).astype(np.uint8)
        else:
            cleaned_src = clean_stage_frame_dynamic(
                frame_src, sec_fl, global_f_idx, fps, timeline_data, bg_clean_ref, clean_ready_pool
            )
            fg_1080 = enhance_foreground_1080p(cleaned_src).astype(np.float32)
            comp = (fg_1080 * stage_alpha_1080 + bg_float * (1.0 - stage_alpha_1080)).astype(np.uint8)

        final_frame = ui_renderer.render_ui(comp, sec_fl)
        proc.stdin.write(final_frame.tobytes())

        if (local_f + 1) % 300 == 0 or (local_f + 1) == total_f:
            elapsed = time.time() - t0
            fps_rate = (local_f + 1) / max(1e-3, elapsed)
            print(
                f"[Render] {local_f+1}/{total_f} frames ({sec_fl:.1f}s) | {fps_rate:.1f} fps | elapsed {elapsed:.1f}s",
                flush=True,
            )

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"[Render] Completed {out_mp4} in {time.time() - t0:.1f}s", flush=True)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Workout Video Remaster Pipeline (Dynamic Template)")
    parser.add_argument("--test-10s", action="store_true", help="Render 10s test slice and extract self-check keyframes")
    parser.add_argument("--test-start", type=float, default=None, help="Start second for 10s test slice (auto-computed if omitted)")
    parser.add_argument("--full", action="store_true", help="Render full remastered 1080P video")
    parser.add_argument("--video", type=str, default=None, help="Input source video path (overrides timeline.json)")
    parser.add_argument("--timeline", type=str, default=str(DEFAULT_TIMELINE), help="Path to timeline.json config")
    parser.add_argument("--bg", type=str, default=None, help="Custom 1080P background image path")
    parser.add_argument("--bgm", type=str, default=None, help="Custom BGM audio path")
    parser.add_argument("--output", type=str, default=None, help="Custom output MP4 path")
    args = parser.parse_args()

    timeline_path = Path(args.timeline).resolve()
    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline config not found at: {timeline_path}")

    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    work_dir = timeline_path.parent
    test_dir = work_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path_str = args.video or timeline_data.get("video_info", {}).get("source_path")
    if not video_path_str or not os.path.exists(video_path_str):
        raise FileNotFoundError(f"Source video not found: {video_path_str}")
    video_path = Path(video_path_str).resolve()

    # 1. Compute dynamic motion envelope
    envelope = compute_motion_envelope(video_path, timeline_data, work_dir)

    # 2. Build dynamic clean reference, ready pool, and peripheral background
    bg_clean_ref, clean_ready_pool, bg_peripheral_1080, stage_alpha_1080 = prepare_stage_assets(
        video_path, timeline_data, envelope, work_dir, bg_arg=args.bg
    )

    # 3. Master audio track
    total_dur = float(timeline_data.get("video_info", {}).get("total_duration_sec", 335.0))
    master_wav = build_full_master_audio(timeline_data, total_dur, work_dir, bgm_arg=args.bgm)

    if args.test_10s:
        test_out = test_dir / "test_10s.mp4"
        blocks = timeline_data.get("blocks", [])

        # Auto-compute test window covering Rest 1 transition into Action 2
        if args.test_start is not None:
            t_start = float(args.test_start)
        elif len(blocks) >= 2:
            t_start = float(blocks[0].get("rest_start_sec", 33.0)) + 4.0
        else:
            t_start = 10.0
        t_end = min(total_dur, t_start + 10.0)

        print(f"[Test-10s] Rendering 10s slice [{t_start:.1f}s .. {t_end:.1f}s] to {test_out}...", flush=True)
        render_video_segment(
            video_path,
            timeline_data,
            bg_clean_ref,
            clean_ready_pool,
            bg_peripheral_1080,
            stage_alpha_1080,
            master_wav,
            test_out,
            start_sec=t_start,
            end_sec=t_end,
            work_dir=work_dir,
        )

        # Render diagnostic slices for keyframe self-checks:
        intro_sec = float(timeline_data.get("intro", {}).get("duration_sec", 0.0))
        if intro_sec > 0:
            aux_intro = test_dir / "test_intro_slice.mp4"
            render_video_segment(
                video_path, timeline_data, bg_clean_ref, clean_ready_pool,
                bg_peripheral_1080, stage_alpha_1080, master_wav, aux_intro,
                start_sec=0.0, end_sec=min(total_dur, max(2.5, intro_sec * 0.3)),
                work_dir=work_dir,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-ss", "1.5", "-i", str(aux_intro), "-frames:v", "1", str(test_dir / "kf0_intro.png")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        if len(blocks) >= 1:
            b1 = blocks[0]
            aux_act = test_dir / "test_act_slice.mp4"
            s1 = float(b1["start_sec"])
            e1 = float(b1["end_sec"])
            r1 = float(b1.get("rest_end_sec", e1 + 10.0))
            render_video_segment(
                video_path, timeline_data, bg_clean_ref, clean_ready_pool,
                bg_peripheral_1080, stage_alpha_1080, master_wav, aux_act,
                start_sec=max(0.0, e1 - 2.0), end_sec=min(total_dur, e1 + 3.0),
                work_dir=work_dir,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-ss", "0.5", "-i", str(aux_act), "-frames:v", "1", str(test_dir / "kf1_action.png")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-ss", "3.0", "-i", str(aux_act), "-frames:v", "1", str(test_dir / "kf2_rest.png")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        subprocess.run(
            ["ffmpeg", "-y", "-ss", "7.0", "-i", str(test_out), "-frames:v", "1", str(test_dir / "kf3_countdown.png")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[Test-10s] Self-check keyframes saved strictly in {test_dir}/", flush=True)
        return

    if args.full:
        out_path = Path(args.output).resolve() if args.output else (out_dir / f"{video_path.stem}_remastered.mp4")
        downloads_out = Path("/Users/guo/Downloads") / f"{video_path.stem}_ailift.mp4"
        if not args.output and downloads_out.parent.exists():
            out_path = downloads_out

        print(f"[Full] Rendering complete {total_dur:.1f}s video to {out_path}...", flush=True)
        render_video_segment(
            video_path,
            timeline_data,
            bg_clean_ref,
            clean_ready_pool,
            bg_peripheral_1080,
            stage_alpha_1080,
            master_wav,
            out_path,
            start_sec=0.0,
            end_sec=total_dur,
            work_dir=work_dir,
        )
        # Also ensure local copy in output directory
        local_copy = out_dir / out_path.name
        if out_path != local_copy:
            shutil.copy2(out_path, local_copy)
        print(f"[Full] Done! Master video ready at: {out_path}", flush=True)


if __name__ == "__main__":
    main()
