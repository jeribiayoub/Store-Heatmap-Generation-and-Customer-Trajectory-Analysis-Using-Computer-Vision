import json
from collections import defaultdict

with open("positions.json") as f:
    positions = json.load(f)

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

MAX_GAP = 45
MAX_DIST = 40

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

used = set()
chains = []

for t in tracks:
    if t["id"] in used:
        continue
    chain = [t]
    used.add(t["id"])
    current = t
    while True:
        candidates = [
            c for c in tracks
            if c["id"] not in used
            and 0 < c["start_frame"] - current["end_frame"] <= MAX_GAP
            and dist(current["end_pos"], c["start_pos"]) <= MAX_DIST
        ]
        if not candidates:
            break
        best = min(candidates, key=lambda c: dist(current["end_pos"], c["start_pos"]))
        chain.append(best)
        used.add(best["id"])
        current = best
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