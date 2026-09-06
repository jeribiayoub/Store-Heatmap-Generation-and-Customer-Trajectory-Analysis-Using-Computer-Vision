import cv2
import numpy as np
import json
from collections import defaultdict

with open("positions_stitched_smoothed.json") as f:
    positions = json.load(f)
with open("chain_gender.json") as f:
    chain_gender = json.load(f)
with open("confirmed_swaps.json") as f:
    confirmed_swaps = json.load(f)
with open("confirmed_reid_matches.json") as f:
    confirmed_reid_matches = json.load(f)
with open("gender_overrides.json") as f:
    gender_overrides = json.load(f)

# --- same identity-relay logic as render_gendered_video.py ---
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

def get_final_gender(chain_id, frame):
    final_chain = get_final_chain(chain_id, frame)
    return gender_overrides.get(str(final_chain), chain_gender.get(str(final_chain), "unknown"))

# --- split positions by final gender ---
man_positions, woman_positions = [], []
for p in positions:
    gender = get_final_gender(p["track_id"], p["frame"])
    if gender == "man":
        man_positions.append(p)
    elif gender == "woman":
        woman_positions.append(p)

print(f"Man points: {len(man_positions)}  |  Woman points: {len(woman_positions)}  |  "
      f"Unknown/excluded: {len(positions) - len(man_positions) - len(woman_positions)}")

# --- reusable full-coverage heatmap renderer (Step 24 style) ---
blueprint = cv2.imread("blueprint.png")
floor_mask = cv2.imread("floor_mask.png", cv2.IMREAD_GRAYSCALE)
bp_h, bp_w = blueprint.shape[:2]

def render_heatmap(point_list, output_path):
    heatmap = np.zeros((bp_h, bp_w), dtype=np.float32)
    for p in point_list:
        x, y = int(p["x"]), int(p["y"])
        if 0 <= x < bp_w and 0 <= y < bp_h:
            heatmap[y, x] += 1

    heatmap_smooth = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=20, sigmaY=20)

    floor_values = heatmap_smooth[floor_mask > 0]
    vmax = floor_values.max() if floor_values.max() > 0 else 1.0

    norm = np.clip(heatmap_smooth / vmax * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    result = blueprint.copy()
    result[floor_mask > 0] = color[floor_mask > 0]

    cv2.imwrite(output_path, result)
    print(f"Saved {output_path} (max density: {vmax:.1f})")

render_heatmap(man_positions, "heatmap_men.jpg")
render_heatmap(woman_positions, "heatmap_women.jpg")