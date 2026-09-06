import cv2
import numpy as np
import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)
with open("confirmed_swaps.json") as f:
    confirmed_swaps = json.load(f)
with open("confirmed_reid_matches.json") as f:
    confirmed_reid_matches = json.load(f)
with open("zones.json") as f:
    zones = json.load(f)
with open("zone_percentages.json") as f:
    zone_percentages = json.load(f)

img = cv2.imread("clean_background.jpg")

# --- same identity-correction logic as render_gendered_video.py ---
def build_identity_relay(confirmed_swaps):
    current_identity = {}
    raw_chain_history = defaultdict(list)

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

reid_merge_map = {m["chain_b"]: m["chain_a"] for m in confirmed_reid_matches}

def get_final_chain(chain_id, frame):
    effective = get_effective_chain(chain_id, frame)
    return reid_merge_map.get(effective, effective)

# --- group foot points by FINAL corrected chain, in frame order ---
points_by_chain = defaultdict(list)
for b in raw_boxes:
    raw_chain = track_to_chain.get(str(b["track_id"]))
    if raw_chain is None:
        continue
    final_chain = get_final_chain(raw_chain, b["frame"])
    foot_x = (b["x1"] + b["x2"]) / 2
    foot_y = b["y2"]
    points_by_chain[final_chain].append((b["frame"], foot_x, foot_y))

for cid in points_by_chain:
    points_by_chain[cid].sort(key=lambda p: p[0])

def color_for_id(chain_id):
    np.random.seed(chain_id)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))

for cid, pts in points_by_chain.items():
    color = color_for_id(cid)
    coords = [(int(x), int(y)) for _, x, y in pts]
    for i in range(1, len(coords)):
        cv2.line(img, coords[i-1], coords[i], color, 2)

# --- zones: transparent color fill scaled by percentage (blue=low -> red=high) ---
pcts = list(zone_percentages.values())
min_pct, max_pct = min(pcts), max(pcts)
pct_range = max_pct - min_pct if max_pct > min_pct else 1

overlay = img.copy()
for z in zones:
    pct = zone_percentages.get(z["name"], 0)
    norm = (pct - min_pct) / pct_range  # 0 = lowest zone, 1 = highest zone
    color_val = np.array([[int(norm * 255)]], dtype=np.uint8)
    bgr_color = cv2.applyColorMap(color_val, cv2.COLORMAP_JET)[0][0]
    cv2.rectangle(overlay, (z["x1"], z["y1"]), (z["x2"], z["y2"]), bgr_color.tolist(), -1)

ALPHA = 0.25  # zone fill transparency
img = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)

# --- zone borders + percentage labels, drawn AFTER blending so they stay crisp ---
for z in zones:
    #cv2.rectangle(img, (z["x1"], z["y1"]), (z["x2"], z["y2"]), (0, 0, 0), 2)
    pct = zone_percentages.get(z["name"], 0)
    label = f"{pct:.2f}%"
    text_x = (z["x1"] + z["x2"]) // 2 - 30
    text_y = (z["y1"] + z["y2"]) // 2
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)

cv2.imwrite("zone_trajectory_view.jpg", img)
print("Saved zone_trajectory_view.jpg")