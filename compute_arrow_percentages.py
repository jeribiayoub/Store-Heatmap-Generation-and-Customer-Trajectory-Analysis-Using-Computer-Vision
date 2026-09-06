import json
import numpy as np
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)
with open("confirmed_swaps.json") as f:
    confirmed_swaps = json.load(f)
with open("confirmed_reid_matches.json") as f:
    confirmed_reid_matches = json.load(f)
with open("arrows.json") as f:
    arrows = json.load(f)

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

# --- build each final chain's ordered point list ---
points_by_chain = defaultdict(list)
for b in raw_boxes:
    raw_chain = track_to_chain.get(str(b["track_id"]))
    if raw_chain is None:
        continue
    final_chain = get_final_chain(raw_chain, b["frame"])
    foot_x = (b["x1"] + b["x2"]) / 2
    foot_y = b["y2"]
    points_by_chain[final_chain].append((b["frame"], foot_x, foot_y))

MIN_LOCAL_DISPLACEMENT = 15  # ignore tracks that barely move near an arrow
MAX_ANGLE_DIFF = 45     # degrees - how close a client's direction must be to an arrow to count
MAX_ARROW_DISTANCE = 32  # max pixel distance from a track point to an arrow segment
LOCAL_WINDOW = 3        # use a small window around the closest approach for direction

arrow_vectors = []
for a in arrows:
    sx, sy = a["points"][0]
    ex, ey = a["points"][-1]
    arrow_vectors.append({
        "name": a["name"],
        "start": np.array([sx, sy], dtype=float),
        "end": np.array([ex, ey], dtype=float),
        "vec": np.array([ex - sx, ey - sy], dtype=float),
    })

def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-3 or n2 < 1e-3:
        return None
    cos_sim = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_sim))


def point_to_segment_metrics(point, start, end):
    segment = end - start
    segment_len_sq = float(np.dot(segment, segment))
    if segment_len_sq < 1e-6:
        return float(np.linalg.norm(point - start)), 0.0

    t = float(np.dot(point - start, segment) / segment_len_sq)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection)), t

arrow_counts = {a["name"]: 0 for a in arrows}
unmatched = 0
total_clients = 0
passed_by_any_arrow = 0

for cid, pts in points_by_chain.items():
    pts.sort(key=lambda p: p[0])
    if len(pts) < 2:
        continue
    track_points = [np.array([x, y], dtype=float) for _, x, y in pts]

    total_clients += 1

    best_match = None
    best_angle = 999
    best_distance = 999
    for av in arrow_vectors:
        near_indices = []
        near_distances = []
        for index, pt in enumerate(track_points):
            dist, _ = point_to_segment_metrics(pt, av["start"], av["end"])
            if dist <= MAX_ARROW_DISTANCE:
                near_indices.append(index)
                near_distances.append(dist)

        if not near_indices:
            continue

        lo = max(0, near_indices[0] - LOCAL_WINDOW)
        hi = min(len(track_points) - 1, near_indices[-1] + LOCAL_WINDOW)
        local_vec = track_points[hi] - track_points[lo]

        if np.linalg.norm(local_vec) < MIN_LOCAL_DISPLACEMENT:
            continue

        diff = angle_between(local_vec, av["vec"])
        if diff is None:
            continue

        angle_diff = min(diff, 180 - diff)
        if angle_diff > MAX_ANGLE_DIFF:
            continue

        closest_distance = min(near_distances)

        if angle_diff < best_angle or (angle_diff == best_angle and closest_distance < best_distance):
            best_angle = angle_diff
            best_distance = closest_distance
            best_match = av["name"]

    if best_match is not None:
        arrow_counts[best_match] += 1
        passed_by_any_arrow += 1
    else:
        unmatched += 1

matched_total = passed_by_any_arrow  # denominator = only people who passed by some arrow

arrow_percentages = {}
for name, count in arrow_counts.items():
    pct = (count / matched_total * 100) if matched_total > 0 else 0
    arrow_percentages[name] = round(pct, 2)
    print(f"{name}: {count} clients -> {pct:.2f}%")

print(f"\nUnmatched (excluded from percentages): {unmatched}")
print(f"Matched clients (used as the 100% base): {matched_total}")

with open("arrow_percentages.json", "w") as f:
    json.dump(arrow_percentages, f, indent=2)