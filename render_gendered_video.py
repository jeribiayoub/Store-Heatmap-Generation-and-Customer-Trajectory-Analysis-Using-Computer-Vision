import cv2
import numpy as np
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)
with open("chain_gender.json") as f:
    chain_gender = json.load(f)
with open("confirmed_swaps.json") as f:
    confirmed_swaps = json.load(f)
with open("gender_overrides.json") as f:
    gender_overrides = json.load(f)

H_direct = np.load("video_to_blueprint.npy")

def video_to_blueprint(px, py):
    p = np.array([px, py, 1.0])
    bp = H_direct @ p
    return bp[0]/bp[2], bp[1]/bp[2]

GENDER_COLOR = {
    "man": (255, 0, 0),
    "woman": (203, 92, 255),
    "unknown": (128, 128, 128),
}

def build_identity_relay(confirmed_swaps):
    current_identity = {}
    raw_chain_history = defaultdict(list)  # raw_chain -> [(frame, new_identity), ...]

    def get_current(chain):
        return current_identity.get(chain, chain)

    for s in sorted(confirmed_swaps, key=lambda s: s["frame"]):
        a, b, f = s["chain_a"], s["chain_b"], s["frame"]
        id_a, id_b = get_current(a), get_current(b)
        current_identity[a] = id_b
        current_identity[b] = id_a
        raw_chain_history[a].append((f, id_b))
        raw_chain_history[b].append((f, id_a))

    return raw_chain_history

RAW_CHAIN_HISTORY = build_identity_relay(confirmed_swaps)

def get_effective_chain(raw_chain_id, frame):
    result = raw_chain_id
    for f_from, ident in sorted(RAW_CHAIN_HISTORY.get(raw_chain_id, [])):
        if frame > f_from:
            result = ident
        else:
            break
    return result

with open("confirmed_reid_matches.json") as f:
    confirmed_reid_matches = json.load(f)

reid_merge_map = {}
for m in confirmed_reid_matches:
    reid_merge_map[m["chain_b"]] = m["chain_a"]  # chain_b becomes chain_a's identity

def get_final_chain(chain_id, frame):
    effective = get_effective_chain(chain_id, frame)
    return reid_merge_map.get(effective, effective)

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

blueprint = cv2.imread("blueprint.png")
bp_h, bp_w = blueprint.shape[:2]

cap = cv2.VideoCapture("data/3.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale = vid_h / bp_h
bp_resized_w = int(bp_w * scale)
out_w = vid_w + bp_resized_w
out = cv2.VideoWriter("comparison_final.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, vid_h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    canvas_bp = cv2.resize(blueprint, (bp_resized_w, vid_h)).copy()

    for b in boxes_by_frame.get(frame_idx, []):
        raw_chain = track_to_chain.get(str(b["track_id"]))
        if raw_chain is None:
            continue
        effective_chain = get_final_chain(raw_chain, frame_idx)
        gender = gender_overrides.get(str(effective_chain), chain_gender.get(str(effective_chain), "unknown"))
        color = GENDER_COLOR[gender]

        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{gender} c{effective_chain}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        foot_x, foot_y = (x1 + x2) / 2, y2
        cv2.circle(frame, (int(foot_x), int(foot_y)), 6, color, -1)

        bx, by = video_to_blueprint(foot_x, foot_y)
        if 0 <= bx < bp_w and 0 <= by < bp_h:
            px, py = int(bx * scale), int(by * scale)
            cv2.circle(canvas_bp, (px, py), 8, color, -1)

    combined = np.hstack([frame, canvas_bp])
    out.write(combined)

    frame_idx += 1
    if frame_idx % 1000 == 0:
        print(f"  {frame_idx}/{total_frames} frames rendered")

cap.release()
out.release()
print("Done. Wrote", frame_idx, "comparison_final.mp4")