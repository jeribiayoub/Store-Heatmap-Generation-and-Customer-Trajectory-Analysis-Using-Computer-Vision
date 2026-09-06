import json
from collections import defaultdict

with open("positions_stitched.json") as f:
    positions = json.load(f)

# group by track_id, keep frame order
by_track = defaultdict(list)
for p in positions:
    by_track[p["track_id"]].append(p)

for track_id in by_track:
    by_track[track_id].sort(key=lambda p: p["frame"])

ALPHA = 0.3  # smoothing strength: lower = smoother but more lag, higher = less smoothing

smoothed = []
for track_id, pts in by_track.items():
    ema_x, ema_y = None, None
    for p in pts:
        if ema_x is None:
            # first point of this track: no previous value to blend with
            ema_x, ema_y = p["x"], p["y"]
        else:
            ema_x = ALPHA * p["x"] + (1 - ALPHA) * ema_x
            ema_y = ALPHA * p["y"] + (1 - ALPHA) * ema_y

        smoothed.append({
            "frame": p["frame"],
            "track_id": track_id,
            "x": ema_x,
            "y": ema_y,
            "gender": p.get("gender", "unknown"),
        })

smoothed.sort(key=lambda p: p["frame"])  # restore chronological order for downstream scripts

with open("positions_stitched_smoothed.json", "w") as f:
    json.dump(smoothed, f)

print(f"Smoothed {len(smoothed)} points across {len(by_track)} tracks.")
print(f"Saved to positions_smoothed.json (alpha={ALPHA})")