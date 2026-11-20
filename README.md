# Workout Video Remaster Pipeline Template (1080P Remaster)

A fully automated, 100% data-driven, and dynamically adaptive video remastering pipeline designed to convert raw fitness/workout videos into professional, clean 1080P YouTube workout experiences with dynamic motion envelope staging, same-color peripheral blend-in, inpainting overlay scrubbers, synchronized 120BPM electronic BGM, dual-track beeps, Edge-TTS coach voiceovers, and modern English HUD cards.

---

## 1. System Requirements & Environment

- **OS**: macOS (Apple Silicon with MPS acceleration recommended) or Linux with CUDA.
- **Python Environment**:
  ```bash
  /Users/guo/.venv/bin/python
  ```
- **Dependencies**:
  - `torch`, `torchvision` (with MPS or CUDA)
  - `sam2` (Segment Anything Model 2.1)
  - `opencv-python`, `numpy`, `pillow`
  - `edge-tts` (Microsoft Neural TTS)
  - `ffmpeg` (CLI installed and accessible in `PATH`)

---

## 2. Dynamic Data-Driven Architecture (`timeline.json`)

The pipeline does not hardcode workout durations, action names, exercise counts, or layout coordinates. Everything is dynamically loaded and computed from a JSON file.

### Complete `timeline.json` Structure & Annotations

```jsonc
{
  "video_info": {
    "source_path": "/path/to/source_video.mp4", // Path to source video
    "source_resolution": "1280x720",            // Source resolution
    "target_resolution": "1920x1080",           // Target output resolution
    "fps": 30,                                  // Frame rate
    "total_duration_sec": 335,                  // Total duration of remastered video
    "title_zh": "5分钟羽毛球核心训练",             // Original Chinese title
    "title_en": "5-Minute Badminton CORE Workout" // English title displayed on HUD
  },
  // Optional clean regions for overlay scrubbing (uses safe dynamic defaults if omitted)
  "clean_regions": {
    "top_bar": {
      "y_max": 274,                             // Everything above y_max is replaced by sampled background
      "description": "Original Chinese menu header and top-center timer zone"
    },
    "bottom_bar": {
      "y_min": 594,                             // Everything below y_min is replaced by sampled background
      "description": "Original yellow Chinese subtitles zone"
    },
    "center_countdown": {
      "x0": 450,
      "y0": 274,
      "x1": 785,
      "y1": 468,                                // Bounding box of giant translucent countdown numbers
      "description": "Center giant translucent white countdown digits and English text"
    },
    "opening_card": {                           // Optional: opening translucent title overlay (0..3s)
      "start_sec": 0.0,
      "end_sec": 3.0,
      "x0": 275,
      "y0": 226,
      "x1": 1004,
      "y1": 487,
      "description": "Translucent white title card 5分钟核心训练"
    },
    "outro_text": {                             // Optional: outro calligraphy or watermark
      "start_sec": 316.8,
      "end_sec": 326.8,
      "x0": 495,
      "y0": 335,
      "x1": 780,
      "y1": 438,
      "description": "Outro Chinese calligraphy 谢谢观看 and red seal above the coach"
    }
  },
  // Optional Intro configuration (set duration_sec to 0 if there is no intro)
  "intro": {
    "start_sec": 0,
    "end_sec": 13,
    "duration_sec": 13,
    "name_zh": "开场准备与倒计时",
    "name_en": "Intro & Ready Countdown"
  },
  // Dynamic list of workout exercises (any number of blocks supported)
  "blocks": [
    {
      "id": 1,
      "name_zh": "死虫子",
      "name_en": "Dead Bug Core Activation",
      "start_sec": 13,
      "end_sec": 33,
      "duration_sec": 20,
      "rest_start_sec": 33,
      "rest_end_sec": 46,
      "rest_duration_sec": 13,
      "next_name_en": "Seated Knee Touch",
      "coach_cue_en": "Keep your lower back pressed firmly into the mat! Stabilize your trunk for overhead smashes."
    },
    {
      "id": 2,
      "name_zh": "坐姿摸膝",
      "name_en": "Seated Knee Touch",
      "start_sec": 46,
      "end_sec": 66,
      "duration_sec": 20,
      "rest_start_sec": 66,
      "rest_end_sec": 79,
      "rest_duration_sec": 13,
      "next_name_en": "Seated Crunch",
      "coach_cue_en": "Engage your upper abs! Control every reach and keep breathing steady."
    }
  ],
  // Optional Outro configuration
  "outro": {
    "start_sec": 317,
    "end_sec": 335,
    "duration_sec": 18,
    "name_zh": "训练完成与致谢",
    "name_en": "Cool Down & Workout Complete"
  }
}
```

---

## 3. CLI Command Reference

All scripts must be executed using `/Users/guo/.venv/bin/python`.

### Parameter Overview
| Argument | Type | Description |
|---|---|---|
| `--timeline` | `str` | Path to workout timeline JSON (default: `workout_remaster/timeline.json`) |
| `--video` | `str` | Path to source video (overrides `video_info.source_path` in timeline JSON) |
| `--test-10s` | `flag` | Renders a 10s slice and outputs 4 self-check keyframes into `workout_remaster/test/` |
| `--test-start`| `float`| Starting second for `--test-10s` (auto-detected if omitted) |
| `--full` | `flag` | Renders the complete 1080P remastered video |
| `--output` | `str` | Custom output path for final `.mp4` video |
| `--bg` | `str` | Optional custom 1080P background image |
| `--bgm` | `str` | Optional custom BGM WAV/MP3 track |

### Standard Execution Flow

1. **Run 10-Second Self-Check**:
   ```bash
   /Users/guo/.venv/bin/python workout_remaster/run_pipeline.py --timeline workout_remaster/timeline.json --test-10s
   ```
   Inspect the extracted keyframes in `workout_remaster/test/`:
   - `kf0_intro.png`: Opening ready posture, solid coach head/body.
   - `kf1_action.png`: Active exercise with zero limb ghosting/flicker.
   - `kf2_rest.png`: Rest period transition and UP NEXT title display.
   - `kf3_countdown.png`: 3-2-1 final countdown with timer highlight.

2. **Run Full 1080P Render**:
   ```bash
   /Users/guo/.venv/bin/python workout_remaster/run_pipeline.py --timeline workout_remaster/timeline.json --full
   ```
