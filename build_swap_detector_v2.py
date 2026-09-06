import json
import numpy as np
from collections import defaultdict

with open("positions_stitched_smoothed.json") as f:
    positions = json.load(f)
with open("chain_signatures.json") as f:
    chain_signatures = json.load(f)

by_chain = defaultdict(list)
for p in positions:
    by_chain[p["track_id"]].append(p)
for cid in by_chain:
    by_chain[cid].sort(key=lambda p: p["frame"])

by_frame = defaultdict(list)
for cid, pts in by_chain.items():
    for p in pts:
        by_frame[p["frame"]].append((cid, p["x"], p["y"]))

PROXIMITY_PX = 30
WINDOW = 10

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
        return None
    cos_sim = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_sim))

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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

    motion_verdict = None
    if va_before is not None and va_after is not None and vb_before is not None and vb_after is not None:
        noswap_total = angle_diff(va_before, va_after) + angle_diff(vb_before, vb_after)
        swap_total = angle_diff(vb_before, va_after) + angle_diff(va_before, vb_after)
        motion_verdict = "swap" if swap_total < noswap_total - 20 else "noswap"

    appearance_verdict = None
    sig_a = chain_signatures.get(str(cid_a))
    sig_b = chain_signatures.get(str(cid_b))
    if sig_a is not None and sig_b is not None:
        same_pair_sim = cosine_sim(sig_a, sig_a)  # placeholder, always 1 - real check below
        cross_sim = cosine_sim(sig_a, sig_b)
        # if A's overall signature looks MORE like B's than like itself would (impossible to check post-hoc),
        # instead: if A and B are already very similar-looking people, appearance can't help distinguish them
        appearance_verdict = "ambiguous" if cross_sim > 0.9 else "distinguishable"

    results.append({
        "chain_a": cid_a, "chain_b": cid_b, "frame": frame,
        "motion_verdict": motion_verdict,
        "appearance_similarity_between_them": round(cosine_sim(sig_a, sig_b), 3) if sig_a and sig_b else None,
        "appearance_verdict": appearance_verdict,
    })

with open("swap_events_v2.json", "w") as f:
    json.dump(results, f, indent=2)

for r in results:
    print(r)