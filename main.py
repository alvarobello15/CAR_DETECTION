import cv2
import imutils
import time
import numpy as np
from ultralytics import YOLO
from reader_frames import FileVideoStream
from centroidtracker import CentroidTracker

# ---------------- CONFIGURACIÓ ----------------
VIDEO_PATH = "output6.mp4"
WIDTH = 400
TARGET_CLASSES = [2]  # coches
QUEUE_SIZE = 32
YOLO_EVERY_N = 5
MIN_CONF = 0.3

# ---------------- LÍNIA DE REFERÈNCIA ----------------
LINE_REF = 525
OFFSET = 5
EXCLUSION_ZONE = (0, 0, 400, 180)

# ---------------- FUNCIONS AUXILIARS ----------------
def in_exclusion(bbox, zone):
    x, y, w, h = bbox
    zx1, zy1, zx2, zy2 = zone
    cx, cy = x + w // 2, y + h // 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

def suprimir_detecciones_solapades(detecciones, iou_thresh=0.6):
    if len(detecciones) <= 1:
        return detecciones
    keep = []
    detecciones = sorted(detecciones, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    for b1 in detecciones:
        suprimir = False
        for b2 in keep:
            x1, y1, x2, y2 = b1
            x3, y3, x4, y4 = b2
            xi1, yi1 = max(x1, x3), max(y1, y3)
            xi2, yi2 = min(x2, x4), min(y2, y4)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union = (x2 - x1)*(y2 - y1) + (x4 - x3)*(y4 - y3) - inter
            iou = inter / union if union > 0 else 0
            if iou > iou_thresh:
                suprimir = True
                break
        if not suprimir:
            keep.append(b1)
    return keep

# ---------------- INICIALITZACIÓ ----------------
print("Cargando modelo YOLOv8n...")
model = YOLO("yolov8n.pt")
model.overrides['conf'] = MIN_CONF
model.overrides['max_det'] = 50

ct = CentroidTracker(maxDisappeared=40)
fvs = FileVideoStream(VIDEO_PATH, queueSize=QUEUE_SIZE).start()
time.sleep(0.0)

frame_count = 0
count_up = 0
count_down = 0
object_last_y = {}
last_detections = []

# ---------------- BUCLE PRINCIPAL ----------------
while True:
    if not fvs.more():
        if fvs.stopped:
            break
        continue

    frame = fvs.read()
    if frame is None:
        continue

    frame = imutils.resize(frame, width=WIDTH)
    frame_count += 1
    rects = []

    # --- YOLO cada N frames ---
    if frame_count % YOLO_EVERY_N == 0:
        results = model(frame, verbose=False, device="cpu")[0]
        boxes = results.boxes
        new_dets = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                if int(cls[i]) in TARGET_CLASSES and conf[i] > MIN_CONF:
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    bbox = (x1, y1, x2, y2)
                    if not in_exclusion((x1, y1, x2 - x1, y2 - y1), EXCLUSION_ZONE):
                        new_dets.append(bbox)
        last_detections = suprimir_detecciones_solapades(new_dets, 0.6)

    rects = last_detections

    # --- Dibuixar les deteccions YOLO ---
    for (x1, y1, x2, y2) in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "car", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Actualitzar el tracker ---
    objects = ct.update(rects,frame)

    # --- Recórrer objectes trackejats ---
    for (objectID, centroid) in objects.items():
        cX, cY = centroid

        # Comprovar travessament línia
        if objectID in object_last_y:
            prevY = object_last_y[objectID]
            if prevY < LINE_REF <= cY:
                count_down += 1
            elif prevY > LINE_REF >= cY:
                count_up += 1
        object_last_y[objectID] = cY

        # Dibuixar ID i centroid
        cv2.circle(frame, (cX, cY), 4, (255, 0, 0), -1)
        cv2.putText(frame, f"ID:{objectID}", (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # --- Dibuixar línia i info ---
    overlay = frame.copy()
    cv2.line(overlay, (0, LINE_REF), (WIDTH, LINE_REF), (0, 0, 255), 2)
    cv2.rectangle(overlay, EXCLUSION_ZONE[:2], EXCLUSION_ZONE[2:], (128, 128, 128), -1)
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    cv2.putText(frame, f"UP: {count_up}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"DOWN: {count_down}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {len(objects)}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("YOLO + Centroid Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fvs.stop()
cv2.destroyAllWindows()

print("\n=== RESULTATS FINALS ===")
print(f"UP (abajo→arriba): {count_up}")
print(f"DOWN (arriba→abajo): {count_down}")
print(f"TOTAL: {count_up + count_down}")
print("===========================")

