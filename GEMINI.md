# AI Collaboration Protocol & Engineering Rules (GEMINI.md)

This file defines the strict engineering guidelines and operating rules for AI models (Gemini, Claude, GPT) working on the `workout_remaster` project.

---

## 1. Zero-Regression Core Pipeline Reuse Rule
- **NEVER** rewrite, replace, or break the core compositing architecture in `workout_remaster/run_pipeline.py`.
- The following battle-tested modules in `run_pipeline.py` are strictly protected:
  1. **Dynamic Motion Envelope Stage**: Auto-scanning SAM 2 global union bounding box with safety margins instead of frame-by-frame jagged matting.
  2. **Seamless Same-Color Peripheral Blend-In**: Dynamic sampling of floor/wall color bands, multi-scale matte blur, vignette, and feathered stage transition.
  3. **Head & Torso Anti-Decapitation Logic**: Solid ready-pose frame pool and profile-aware protection during countdown removal.
  4. **Non-Overlapping English HUD**: Dynamic badge pill sizing, fitted/multi-line exercise title wrapping, and segmented progress bar spacing.

---

## 2. Standardization Workflow for New Videos
Whenever you receive a new workout video:
1. **Never edit `run_pipeline.py` to hardcode numbers**:
   - Inspect the new video's duration, resolution, athlete posture, and text layout.
   - Check where original Chinese subtitles, top menus, and countdown digits appear.
2. **Generate a clean `timeline_<name>.json`**:
   - Define exact timestamps for exercises, rests, optional intro, and outro.
   - Configure custom `clean_regions` if the subtitle or timer positions differ from default.
3. **Execute 10s Self-Check**:
   ```bash
   /Users/guo/.venv/bin/python workout_remaster/run_pipeline.py --timeline workout_remaster/timeline_<name>.json --test-10s
   ```
4. **Visually Inspect Keyframes in `workout_remaster/test/`**:
   - Check `kf0_intro.png`, `kf1_action.png`, `kf2_rest.png`, `kf3_countdown.png`.
   - Confirm: zero text collision, no limb flickering, solid head, seamless background.
5. **Execute Full Render**:
   ```bash
   /Users/guo/.venv/bin/python workout_remaster/run_pipeline.py --timeline workout_remaster/timeline_<name>.json --full
   ```

---

## 3. Storage & Path Isolation Rule
- **Workspace-Only Output**: All intermediate debug images, keyframe test frames, test videos, and render logs MUST be written to `workout_remaster/test/`.
- **STRICT PROHIBITION**: Never write files or temporary artifacts into `/tmp`, `/private/tmp`, or paths outside the project workspace.
