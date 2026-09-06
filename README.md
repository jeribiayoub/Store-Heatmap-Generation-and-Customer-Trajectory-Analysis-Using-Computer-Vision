# Store Traffic Analytics from CCTV Footage

A computer vision pipeline that turns a single fixed CCTV camera feed from a
retail store into customer traffic analytics: detecting and tracking
customers, classifying gender, and mapping their movement onto a 2D floor
plan of the store to generate heatmaps and traffic visualizations.

## What it does

Given a store's CCTV footage and a hand-drawn blueprint of the store layout,
the pipeline produces:

- **Gender-split heatmaps** — two floor-plan heatmaps (one for men, one for
  women) showing where each group spends the most time.
- **A camera-view heatmap** — the same density information overlaid directly
  on a clean (people-free) photo of the store from the camera's own angle.
- **Zone traffic breakdown** — user-defined rectangular zones on the store
  photo, each colored by its share of overall foot traffic, with customer
  trajectories drawn on top.
- **Directional flow arrows** — curved arrows showing what percentage of
  customers moved in each general direction through the store.

## Pipeline overview

1. **Detection** — YOLOv8m detects people in each frame.
2. **Tracking** — BoT-SORT with ReID (appearance-based re-identification)
   follows each person across frames, recovering their identity through
   brief occlusions.
3. **Homography calibration** — a one-time-per-camera geometric calibration
   maps pixel positions in the video to positions on the store's 2D
   blueprint.
4. **Gender classification** — CLIP (zero-shot) classifies each detected
   person as man or woman, with confidence-weighted voting across their
   whole tracked visit.
5. **Track stitching** — an appearance-aware merging step reconnects
   fragments of the same person's visit that got split by occlusion or a
   camera-angle change, using a dedicated person re-identification (ReID)
   model.
6. **Visualization** — final positions and gender labels are rendered onto
   the blueprint (as heatmaps) and onto a clean store photo (as zones and
   directional arrows).

## Requirements

- Python 3.10+
- An NVIDIA GPU is strongly recommended (CPU works but is much slower)

Install dependencies:

```bash
pip install ultralytics opencv-python numpy torch torchvision transformers
pip install boxmot==12.0.2 --no-deps
pip install ftfy gitpython lapx loguru
```

If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch instead
of the CPU-only default — see [pytorch.org](https://pytorch.org) for the
correct command for your CUDA version.

## Project setup (per video)

Each new video needs its own one-time setup before the pipeline can run:

1. Place your video at `data/3.mp4`.
2. Draw a top-down blueprint of the store layout (any image editor) and
   save it as `blueprint.png` in the project root.
3. Calibrate the camera-to-blueprint mapping (see **Usage**, step 1 below) —
   this only needs to be done once per camera/blueprint pair.

## Usage

Run the following in order from the project root, with your virtual
environment activated.

### 1. Calibrate the camera-to-blueprint mapping

```bash
python pick_points.py
```
Click on 2-3 identifiable rectangular fixtures (tables, shelf islands) in
the video frame — 4 corners each, in a consistent order (near-left,
near-right, far-right, far-left).

```bash
python pick_blueprint_points.py
```
Click the same fixtures' corners on `blueprint.png`, in the same order.

Update the point arrays inside `build_direct_transform.py` with the
coordinates you just collected, then run:

```bash
python build_direct_transform.py
```
This produces `video_to_blueprint.npy` — the calibration used by every
downstream step. Check the printed sanity check: predicted and actual
blueprint coordinates should be close for all points.

```bash
python pipeline_test.py
```
Renders a short test clip so you can visually confirm people map to
sensible positions on the blueprint before processing the full video.

### 2. Run the main extraction

```bash
python extract_full_data.py
```
Runs detection, tracking, and gender classification over the full video.

### 3. Stitch tracks and finalize gender labels

```bash
python build_track_signatures.py
python stitch_tracks_v2.py
python smooth_positions.py
python stitch_gender.py
python build_reid_signatures.py
```

### 4. Build the blueprint heatmaps

```bash
python build_floor_mask.py
```
Click several points on clearly empty floor areas of `blueprint.png` — this
separates walkable floor from shelves/fixtures so only the floor gets
colored.

```bash
python build_gender_heatmaps.py
```
Produces `heatmap_men.jpg` and `heatmap_women.jpg`.

### 5. Build the camera-view heatmap

```bash
python build_clean_background.py
```
Generates a people-free photo of the store (`clean_background.jpg`) by
taking the median of many sampled frames.

```bash
python build_camera_view_heatmap.py
```
Produces `heatmap_camera_view.jpg`.

### 6. Build the zone traffic view

```bash
python pick_zones_drag.py
```
Drag out rectangular zones on `clean_background.jpg` and name each one.

```bash
python compute_zone_traffic.py
python build_zone_trajectory_view.py
```
Produces `zone_trajectory_view.jpg` — zones colored by traffic share, with
customer trajectories drawn on top.

### 7. Build the directional flow arrows

```bash
python pick_arrows_curved.py
```
Click a curved path (start to end) for each direction you want to measure,
naming each one.

```bash
python compute_arrow_percentages.py
python build_arrow_view_curved.py
```
Produces `arrow_view_curved.jpg` — curved arrows sized and colored by what
percentage of customers moved in that direction.

## Notes and limitations

- The homography, blueprint, floor mask, zones, and arrows are all specific
  to one camera/store layout and must be redone for each new video.
- Gender classification accuracy depends on footage quality (resolution,
  angle, occlusion) and reflects known limitations of appearance-based
  classification from CCTV — expect it to be a useful approximation, not
  perfectly accurate.
- Detection and tracking accuracy will vary with camera angle, resolution,
  and how crowded the footage is.
