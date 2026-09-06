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

# map: chain_id -> {frame: box}, and chain_id -> sorted list of frames it actually has data in
box_by_chain_frame = defaultdict(dict)
for b in raw_boxes:
    chain_id = track_to_chain.get(str(b["track_id"]))
    if chain_id is not None:
        box_by_chain_frame[chain_id][b["frame"]] = b

chain_frames_sorted = {cid: sorted(frames.keys()) for cid, frames in box_by_chain_frame.items()}

# find candidate encounters (same proximity logic as before)
by_chain_pos = defaultdict(list)
for p in positions:
    by_chain_pos[p["track_id"]].append(p)
by_frame_pos = defaultdict(list)
for cid, pts in by_chain_pos.items():
    for p in pts:
        by_frame_pos[p["frame"]].append((cid, p["x"], p["y"]))

PROXIMITY_PX = 30
seen_pairs = {}
for frame in sorted(by_frame_pos.keys()):
    entries = by_frame_pos[frame]
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            cid_a, xa, ya = entries[i]
            cid_b, xb, yb = entries[j]
            if cid_a == cid_b:
                continue
            if ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5 <= PROXIMITY_PX:
                key = tuple(sorted((cid_a, cid_b)))
                if key not in seen_pairs:
                    seen_pairs[key] = frame

print(f"Found {len(seen_pairs)} candidate encounters.")
encounters = [{"cid_a": a, "cid_b": b, "frame": f} for (a, b), f in seen_pairs.items()]

K = 3              # how many real frames of data we need on each side
MAX_SEARCH = 90     # don't look further than 3 seconds away (at 30fps) - stay relevant to THIS crossing

def find_frames(chain_id, ref_frame, before, k=K, max_search=MAX_SEARCH):
    all_frames = chain_frames_sorted.get(chain_id, [])
    if before:
        candidates = [f for f in all_frames if ref_frame - max_search <= f < ref_frame]
        return candidates[-k:] if len(candidates) >= 1 else []
    else:
        candidates = [f for f in all_frames if ref_frame < f <= ref_frame + max_search]
        return candidates[:k] if len(candidates) >= 1 else []

# figure out exactly which (chain, frame) pairs we need embeddings for
needed_pairs = set()
encounter_frame_refs = []  # per encounter: which specific frames to average for each of the 4 slots
for e in encounters:
    slots = {}
    for label, chain_id, before in [("a_before", e["cid_a"], True), ("a_after", e["cid_a"], False),
                                      ("b_before", e["cid_b"], True), ("b_after", e["cid_b"], False)]:
        frames = find_frames(chain_id, e["frame"], before)
        slots[label] = (chain_id, frames)
        for f in frames:
            needed_pairs.add((chain_id, f))
    encounter_frame_refs.append(slots)

print(f"Need embeddings for {len(needed_pairs)} (chain, frame) pairs.")

needed_frames = defaultdict(list)  # frame -> list of chain_ids needed at that frame
for chain_id, f in needed_pairs:
    needed_frames[f].append(chain_id)

embedding_store = {}  # (chain_id, frame) -> embedding

cap = cv2.VideoCapture("data/3.mp4")
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in needed_frames:
        wanted_chains = set(needed_frames[frame_idx])
        boxes_here = []
        chain_ids_here = []
        for chain_id in wanted_chains:
            b = box_by_chain_frame[chain_id].get(frame_idx)
            if b:
                boxes_here.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                chain_ids_here.append(chain_id)
        if boxes_here:
            embeddings = reid_model.get_features(np.array(boxes_here), frame)
            for cid, emb in zip(chain_ids_here, embeddings):
                embedding_store[(cid, frame_idx)] = emb

    frame_idx += 1
    if frame_idx % 1000 == 0:
        print(f"  {frame_idx} frames scanned")

cap.release()

def avg_slot(chain_id, frames):
    embs = [embedding_store[(chain_id, f)] for f in frames if (chain_id, f) in embedding_store]
    if not embs:
        return None
    v = np.mean(embs, axis=0)
    return v / np.linalg.norm(v)

results = []
for e, slots in zip(encounters, encounter_frame_refs):
    a_before = avg_slot(*slots["a_before"])
    a_after  = avg_slot(*slots["a_after"])
    b_before = avg_slot(*slots["b_before"])
    b_after  = avg_slot(*slots["b_after"])

    missing = [name for name, v in [("a_before", a_before), ("a_after", a_after),
                                      ("b_before", b_before), ("b_after", b_after)] if v is None]
    if missing:
        total_a = len(chain_frames_sorted.get(e["cid_a"], []))
        total_b = len(chain_frames_sorted.get(e["cid_b"], []))
        print(f"Encounter {e['cid_a']}/{e['cid_b']} @ {e['frame']}: missing {missing} | "
              f"chain_a total frames in WHOLE video: {total_a} | chain_b total frames in WHOLE video: {total_b}")
        results.append({**e, "verdict": None, "reason": f"missing {missing}"})
        continue

    noswap_score = float(np.dot(a_before, a_after) + np.dot(b_before, b_after))
    swap_score = float(np.dot(a_before, b_after) + np.dot(b_before, a_after))
    verdict = "likely_swap" if swap_score > noswap_score + 0.3 else "likely_noswap"

    results.append({**e, "verdict": verdict,
                     "noswap_score": round(noswap_score, 3), "swap_score": round(swap_score, 3)})

with open("swap_events_v4.json", "w") as f:
    json.dump(results, f, indent=2)

judged = [r for r in results if r["verdict"] is not None]
flagged = [r for r in judged if r["verdict"] == "likely_swap"]
print(f"\nJudged {len(judged)} of {len(results)} encounters.")
print(f"Flagged {len(flagged)} as likely swaps:")
for r in flagged:
    print(r)