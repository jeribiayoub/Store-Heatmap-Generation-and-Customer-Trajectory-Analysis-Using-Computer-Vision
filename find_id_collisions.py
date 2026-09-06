import json
from collections import defaultdict

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)
with open("confirmed_swaps.json") as f:
    confirmed_swaps = json.load(f)

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

boxes_by_frame = defaultdict(list)
for b in raw_boxes:
    boxes_by_frame[b["frame"]].append(b)

collisions_found = 0
for frame_idx, boxes in boxes_by_frame.items():
    effective_ids_here = []
    for b in boxes:
        raw_chain = track_to_chain.get(str(b["track_id"]))
        if raw_chain is None:
            continue
        eff = get_effective_chain(raw_chain, frame_idx)
        effective_ids_here.append((eff, raw_chain))

    # check for duplicates
    seen = {}
    for eff, raw in effective_ids_here:
        if eff in seen and seen[eff] != raw:
            print(f"COLLISION at frame {frame_idx}: display id {eff} used by BOTH raw chain {seen[eff]} and raw chain {raw}")
            collisions_found += 1
        seen[eff] = raw

print(f"\nTotal collisions found: {collisions_found}")