import json

with open("raw_boxes.json") as f:
    raw_boxes = json.load(f)
with open("zones.json") as f:
    zones = json.load(f)

def get_zone(x, y):
    for z in zones:
        if z["x1"] <= x <= z["x2"] and z["y1"] <= y <= z["y2"]:
            return z["name"]
    return None

zone_counts = {z["name"]: 0 for z in zones}
total_in_zones = 0

for b in raw_boxes:
    foot_x = (b["x1"] + b["x2"]) / 2
    foot_y = b["y2"]
    zone = get_zone(foot_x, foot_y)
    if zone:
        zone_counts[zone] += 1
        total_in_zones += 1

zone_percentages = {}
for name, count in zone_counts.items():
    pct = (count / total_in_zones * 100) if total_in_zones > 0 else 0
    zone_percentages[name] = round(pct, 2)
    print(f"{name}: {count} points -> {pct:.2f}%")

with open("zone_percentages.json", "w") as f:
    json.dump(zone_percentages, f, indent=2)

print(f"\nTotal points inside any zone: {total_in_zones}")
print("Saved zone_percentages.json")