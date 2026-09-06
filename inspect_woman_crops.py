import cv2
import json

with open('raw_boxes.json') as f:
    boxes = json.load(f)
with open('track_gender.json') as f:
    tg = json.load(f)

woman_ids = set(tid for tid, g in tg.items() if g == 'woman')
print("Woman track IDs:", woman_ids)

cap = cv2.VideoCapture('data/3.mp4')
count = 0
for b in boxes:
    if str(b['track_id']) in woman_ids and count < 10:
        cap.set(cv2.CAP_PROP_POS_FRAMES, b['frame'])
        ret, frame = cap.read()
        x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            filename = f"woman_check_{b['track_id']}_{count}.jpg"
            cv2.imwrite(filename, crop)
            count += 1
cap.release()
print(f"Saved {count} crops for inspection")