import json
from collections import defaultdict

with open("track_conf_sums.json") as f:
    conf_sum = json.load(f)
with open("track_to_chain.json") as f:
    track_to_chain = json.load(f)

chain_sum = defaultdict(lambda: {"man": 0.0, "woman": 0.0})
for tid, sums in conf_sum.items():
    chain_id = track_to_chain.get(tid)
    if chain_id is None:
        continue
    chain_sum[chain_id]["man"] += sums["man"]
    chain_sum[chain_id]["woman"] += sums["woman"]

MARGIN = 0.1
chain_gender = {}
for chain_id, sums in chain_sum.items():
    total = sums["man"] + sums["woman"]
    if total == 0:
        continue
    man_share, woman_share = sums["man"]/total, sums["woman"]/total
    if man_share - woman_share > MARGIN:
        chain_gender[chain_id] = "man"
    elif woman_share - man_share > MARGIN:
        chain_gender[chain_id] = "woman"
    else:
        chain_gender[chain_id] = "unknown"

with open("chain_gender.json", "w") as f:
    json.dump(chain_gender, f)

man_n = sum(1 for v in chain_gender.values() if v == "man")
woman_n = sum(1 for v in chain_gender.values() if v == "woman")
unk_n = sum(1 for v in chain_gender.values() if v == "unknown")
print(f"Done. {len(chain_gender)} chains classified. man={man_n}, woman={woman_n}, unknown={unk_n}")