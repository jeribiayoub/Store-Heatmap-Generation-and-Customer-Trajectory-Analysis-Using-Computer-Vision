import json
import numpy as np
from collections import defaultdict

with open("positions_stitched_smoothed.json") as f:
    positions = json.load(f)

# NOTE: "track_id" here is actually your stitched CHAIN id (stitch_tracks.py relabels it)
by_chain = defaultdict(list)
for p in positions:
    by_chain[p["track_id"]].append(p)
for cid in by_chain:
    by_chain[cid].sort(key=lambda p: p["frame"])

by_frame = defaultdict(list)
for cid, pts in by_chain.items():
    for p in pts:
        by_frame[p["frame"]].append((cid, p["x"], p["y"]))

PROXIMITY_PX = 30   # how close (in blueprint pixels) counts as a "crossing" - tune if needed
WINDOW = 10         # frames before/after to estimate direction of motion

def get_velocity(chain_pts, ref_frame, before=True):
    if before:
        window_pts = [p for p in chain_pts if ref_frame - WINDOW <= p["frame"] < ref_frame]
    else:
        window_pts = [p for p in chain_pts if ref_frame < p["frame"] <= ref_frame + WINDOW]
    if len(window_pts) < 2:
        return None
    dx = window_pts[-1]["x"] - window_pts[0]["x"]
    dy = window_pts[-1]["y"] - window_pts[0]["y"]
    return np.array([dx, dy])

def angle_diff(v1, v2):
    if v1 is None or v2 is None:
        return None
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-3 or n2 < 1e-3:
        return None  # too stationary to judge direction
    cos_sim = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_sim))

seen_pairs = {}
for frame in sorted(by_frame.keys()):
    entries = by_frame[frame]
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            cid_a, xa, ya = entries[i]
            cid_b, xb, yb = entries[j]
            if cid_a == cid_b:
                continue
            dist = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
            if dist <= PROXIMITY_PX:
                key = tuple(sorted((cid_a, cid_b)))
                if key not in seen_pairs:
                    seen_pairs[key] = frame

print(f"Found {len(seen_pairs)} candidate close encounters.")

results = []
for (cid_a, cid_b), frame in seen_pairs.items():
    pts_a, pts_b = by_chain[cid_a], by_chain[cid_b]

    va_before = get_velocity(pts_a, frame, before=True)
    va_after  = get_velocity(pts_a, frame, before=False)
    vb_before = get_velocity(pts_b, frame, before=True)
    vb_after  = get_velocity(pts_b, frame, before=False)

    if va_before is None or va_after is None or vb_before is None or vb_after is None:
        continue  # not enough motion data around this encounter to judge

    noswap_total = angle_diff(va_before, va_after) + angle_diff(vb_before, vb_after)
    swap_total = angle_diff(vb_before, va_after) + angle_diff(va_before, vb_after)

    results.append({
        "chain_a": cid_a, "chain_b": cid_b, "frame": frame,
        "noswap_score": round(noswap_total, 1),
        "swap_score": round(swap_total, 1),
        "likely_swap": bool(swap_total < noswap_total - 20)  # require a clear margin, not a marginal call
    })

with open("swap_events.json", "w") as f:
    json.dump(results, f, indent=2)

likely = [r for r in results if r["likely_swap"]]
print(f"Analyzed {len(results)} encounters with enough motion data to judge.")
print(f"Flagged {len(likely)} as likely real ID swaps:")
for r in likely:
    print(r)