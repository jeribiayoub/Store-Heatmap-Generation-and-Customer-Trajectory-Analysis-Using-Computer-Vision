import cv2
import numpy as np
import json

with open("arrows.json") as f:
    arrows = json.load(f)
with open("arrow_percentages.json") as f:
    arrow_percentages = json.load(f)

img = cv2.imread("clean_background.jpg")
overlay = img.copy()

h, w = img.shape[:2]
draw_scale = w / 640  # 640 = reference resolution your original sizes were tuned for

pcts = list(arrow_percentages.values())
min_pct, max_pct = min(pcts), max(pcts)
pct_range = max_pct - min_pct if max_pct > min_pct else 1

MIN_THICKNESS, MAX_THICKNESS = int(4 * draw_scale), int(14 * draw_scale)
ALPHA = 0.6
FONT_SCALE = 0.9 * draw_scale
FONT_THICKNESS_OUTER = max(1, int(3 * draw_scale))
FONT_THICKNESS_INNER = max(1, int(1 * draw_scale))

def catmull_rom(points, n_per_segment=30):
    pts = np.array(points, dtype=float)
    if len(pts) < 3:
        t = np.linspace(0, 1, n_per_segment)
        return pts[0] + t[:, None] * (pts[-1] - pts[0])
    p = np.vstack([pts[0], pts, pts[-1]])
    result = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
        for t in np.linspace(0, 1, n_per_segment, endpoint=False):
            t2, t3 = t*t, t*t*t
            pt = 0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t2 + (-p0+3*p1-3*p2+p3)*t3)
            result.append(pt)
    result.append(p[-2])
    return np.array(result)

for a in arrows:
    name = a["name"]
    pct = arrow_percentages.get(name, 0)
    norm = (pct - min_pct) / pct_range

    color_val = np.array([[int(norm * 255)]], dtype=np.uint8)
    bgr_color = cv2.applyColorMap(color_val, cv2.COLORMAP_JET)[0][0].tolist()
    thickness = int(MIN_THICKNESS + norm * (MAX_THICKNESS - MIN_THICKNESS))

    curve = catmull_rom(a["points"]).astype(int)
    cv2.polylines(overlay, [curve], False, bgr_color, thickness, lineType=cv2.LINE_AA)

    # arrowhead: use the last couple of curve points to get the final direction
    tail_pt = tuple(curve[-6]) if len(curve) > 6 else tuple(curve[0])
    head_pt = tuple(curve[-1])
    cv2.arrowedLine(overlay, tail_pt, head_pt, bgr_color, thickness, tipLength=1.2, line_type=cv2.LINE_AA)

    mid = curve[len(curve)//2]
    label = f"{pct:.0f}%"
    cv2.putText(img, label, (int(mid[0])+10, int(mid[1])-10), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (255,255,255), FONT_THICKNESS_OUTER)
    cv2.putText(img, label, (int(mid[0])+10, int(mid[1])-10), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0,0,0), FONT_THICKNESS_INNER)

result = cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0)
cv2.imwrite("arrow_view_curved.jpg", result)
print("Saved arrow_view_curved.jpg")