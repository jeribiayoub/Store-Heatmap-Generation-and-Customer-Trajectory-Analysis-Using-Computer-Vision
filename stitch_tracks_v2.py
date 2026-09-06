import json
import numpy as np
from collections import defaultdict

with open("positions.json") as f:
    positions = json.load(f)
with open("track_signatures_raw.json") as f:
    track_signatures = json.load(f)  # keys are strings

by_track = defaultdict(list)
for p in positions:
    by_track[p["track_id"]].append(p)
for tid in by_track:
    by_track[tid].sort(key=lambda p: p["frame"])

tracks = []
for tid, pts in by_track.items():
    tracks.append({
        "id": tid,
        "start_frame": pts[0]["frame"], "start_pos": (pts[0]["x"], pts[0]["y"]),
        "end_frame": pts[-1]["frame"], "end_pos": (pts[-1]["x"], pts[-1]["y"]),
        "points": pts,
    })
tracks.sort(key=lambda t: t["start_frame"])

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def appearance_sim(tid_a, tid_b):
    sig_a = track_signatures.get(str(tid_a))
    sig_b = track_signatures.get(str(tid_b))
    if sig_a is None or sig_b is None:
        return None
    a, b = np.array(sig_a), np.array(sig_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Tier 1: tight position/time match, no appearance check needed (same as before)
TIGHT_GAP, TIGHT_DIST = 45, 40
# Tier 2: wider gap/distance, but REQUIRES strong appearance match to accept
WIDE_GAP, WIDE_DIST = 150, 150
APPEARANCE_THRESHOLD = 0.55  # well above the ~0.53 we measured between different people

used = set()
chains = []

for t in tracks:
    if t["id"] in used:
        continue
    chain = [t]
    used.add(t["id"])
    current = t
    while True:
        tight_candidates = [
            c for c in tracks
            if c["id"] not in used
            and 0 < c["start_frame"] - current["end_frame"] <= TIGHT_GAP
            and dist(current["end_pos"], c["start_pos"]) <= TIGHT_DIST
        ]
        if tight_candidates:
            best = min(tight_candidates, key=lambda c: dist(current["end_pos"], c["start_pos"]))
            chain.append(best)
            used.add(best["id"])
            current = best
            continue

        wide_candidates = [
            c for c in tracks
            if c["id"] not in used
            and 0 < c["start_frame"] - current["end_frame"] <= WIDE_GAP
            and dist(current["end_pos"], c["start_pos"]) <= WIDE_DIST
        ]
        wide_candidates_with_sim = []
        for c in wide_candidates:
            sim = appearance_sim(current["id"], c["id"])
            if sim is not None and sim >= APPEARANCE_THRESHOLD:
                wide_candidates_with_sim.append((c, sim))

        if wide_candidates_with_sim:
            best, best_sim = max(wide_candidates_with_sim, key=lambda x: x[1])
            chain.append(best)
            used.add(best["id"])
            current = best
            continue

        break
    chains.append(chain)

print(f"Original tracks: {len(tracks)}  ->  Stitched chains: {len(chains)}")

track_to_chain = {}
for new_id, chain in enumerate(chains):
    for t in chain:
        track_to_chain[t["id"]] = new_id

with open("track_to_chain.json", "w") as f:
    json.dump(track_to_chain, f)

stitched_positions = []
for new_id, chain in enumerate(chains):
    for t in chain:
        for p in t["points"]:
            stitched_positions.append({
                "frame": p["frame"], "track_id": new_id,
                "x": p["x"], "y": p["y"],
                "gender": p.get("gender", "unknown"),
            })

stitched_positions.sort(key=lambda p: p["frame"])
with open("positions_stitched.json", "w") as f:
    json.dump(stitched_positions, f)

print("Saved positions_stitched.json and track_to_chain.json")