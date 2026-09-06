import cv2
import numpy as np
import torch
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.utils import WEIGHTS
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)
with open("positions_stitched_smoothed.json") as f:
    positions = json.load(f)

reid_model = ReidAutoBackend(weights=WEIGHTS / "osnet_x0_25_msmt17.pt", device=torch.device("cuda:0")).model

by_chain_pos = defaultdict(list)
for p in positions:
    by_chain_pos[p["track_id"]].append(p)
for cid in by_chain_pos:
    by_chain_pos[cid].sort(key=lambda p: p["frame"])

by_frame_pos = defaultdict(list)
for cid, pts in by_chain_pos.items():
    for p in pts:
        by_frame_pos[p["frame"]].append((cid, p["x"], p["y"]))

PROXIMITY_PX = 30
WINDOW = 5  # frames before/after each crossing to sample for identity comparison

# find candidate encounters (same logic as before)
seen_pairs = {}
for frame in sorted(by_frame_pos.keys()):
    entries = by_frame_pos[frame]
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

print(f"Found {len(seen_pairs)} candidate encounters.")

encounters = [{"cid_a": a, "cid_b": b, "frame": f} for (a, b), f in seen_pairs.items()]

# figure out which frames we need to visit, and for which (encounter, chain, before/after) slots
needed = defaultdict(list)  # frame -> list of (encounter_idx, chain_id, "before"/"after")
for idx, e in enumerate(encounters):
    for chain_id in (e["cid_a"], e["cid_b"]):
        for f in range(e["frame"] - WINDOW, e["frame"]):
            needed[f].append((idx, chain_id, "before"))
        for f in range(e["frame"] + 1, e["frame"] + WINDOW + 1):
            needed[f].append((idx, chain_id, "after"))

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

# accumulator: (encounter_idx, chain_id, "before"/"after") -> list of embeddings
acc = defaultdict(list)

cap = cv2.VideoCapture("data/3.mp4")
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in needed:
        frame_boxes = boxes_by_frame.get(frame_idx, [])
        if frame_boxes:
            xyxys = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in frame_boxes])
            embeddings = reid_model.get_features(xyxys, frame)
            box_chain_ids = [track_to_chain.get(str(b["track_id"])) for b in frame_boxes]

            for slot in needed[frame_idx]:
                idx, chain_id, when = slot
                for bc_id, emb in zip(box_chain_ids, embeddings):
                    if bc_id == chain_id:
                        acc[(idx, chain_id, when)].append(emb)

    frame_idx += 1
    if frame_idx % 1000 == 0:
        print(f"  {frame_idx} frames scanned")

cap.release()

def avg_emb(idx, chain_id, when):
    vals = acc.get((idx, chain_id, when))
    if not vals:
        return None
    v = np.mean(vals, axis=0)
    return v / np.linalg.norm(v)

results = []
for idx, e in enumerate(encounters):
    a_before = avg_emb(idx, e["cid_a"], "before")
    a_after  = avg_emb(idx, e["cid_a"], "after")
    b_before = avg_emb(idx, e["cid_b"], "before")
    b_after  = avg_emb(idx, e["cid_b"], "after")

    missing = [name for name, v in [("a_before", a_before), ("a_after", a_after),
                                      ("b_before", b_before), ("b_after", b_after)] if v is None]
    if missing:
        print(f"Encounter {e['cid_a']}/{e['cid_b']} @ frame {e['frame']}: missing {missing}")
        results.append({**e, "verdict": None, "reason": "insufficient data"})
        continue

    noswap_score = np.dot(a_before, a_after) + np.dot(b_before, b_after)
    swap_score = np.dot(a_before, b_after) + np.dot(b_before, a_after)

    verdict = "likely_swap" if swap_score > noswap_score + 0.3 else "likely_noswap"
    missing = [name for name, v in [("a_before", a_before), ("a_after", a_after),
                                      ("b_before", b_before), ("b_after", b_after)] if v is None]
    if missing:
        print(f"Encounter {e['cid_a']}/{e['cid_b']} @ frame {e['frame']}: missing {missing}")
    results.append({
        **e, "verdict": verdict,
        "noswap_score": round(float(noswap_score), 3),
        "swap_score": round(float(swap_score), 3),
    })

with open("swap_events_v3.json", "w") as f:
    json.dump(results, f, indent=2)

judged = [r for r in results if r["verdict"] is not None]
flagged = [r for r in judged if r["verdict"] == "likely_swap"]
print(f"\nJudged {len(judged)} of {len(results)} encounters.")
print(f"Flagged {len(flagged)} as likely swaps:")
for r in flagged:
    print(r)