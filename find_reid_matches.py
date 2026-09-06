import json
import numpy as np
from collections import defaultdict

with open("chain_signatures_reid.json") as f:
    chain_signatures = json.load(f)  # keys are strings
with open("positions_stitched_smoothed.json") as f:
    positions = json.load(f)

# figure out each chain's active frame range
chain_range = {}
by_chain = defaultdict(list)
for p in positions:
    by_chain[p["track_id"]].append(p["frame"])
for cid, frames in by_chain.items():
    chain_range[cid] = (min(frames), max(frames))

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

SIM_THRESHOLD = 0.7  # starting point - we'll tune based on what we see

candidates = []
chain_ids = list(chain_signatures.keys())
for i in range(len(chain_ids)):
    for j in range(i + 1, len(chain_ids)):
        cid_a, cid_b = int(chain_ids[i]), int(chain_ids[j])
        if cid_a not in chain_range or cid_b not in chain_range:
            continue

        start_a, end_a = chain_range[cid_a]
        start_b, end_b = chain_range[cid_b]

        # must NOT overlap in time - one has to fully finish before the other starts
        if end_a < start_b:
            gap = start_b - end_a
        elif end_b < start_a:
            gap = start_a - end_b
        else:
            continue  # they overlap in time - can't be the same person

        sim = cosine_sim(chain_signatures[chain_ids[i]], chain_signatures[chain_ids[j]])
        if sim >= SIM_THRESHOLD:
            candidates.append({
                "chain_a": cid_a, "chain_b": cid_b,
                "similarity": round(sim, 3), "frame_gap": gap,
                "a_range": [start_a, end_a], "b_range": [start_b, end_b],
            })

candidates.sort(key=lambda c: -c["similarity"])

with open("reid_match_candidates.json", "w") as f:
    json.dump(candidates, f, indent=2)

print(f"Found {len(candidates)} candidate long-gap matches (similarity >= {SIM_THRESHOLD}):")
for c in candidates:
    print(c)