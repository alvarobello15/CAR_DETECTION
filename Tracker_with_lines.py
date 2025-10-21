import cv2
import imutils
import time
import numpy as np
from ultralytics import YOLO
from reader_frames import FileVideoStream

# ---------------- CONFIG ----------------
VIDEO_PATH = "output7.mp4"
WIDTH = 400
TARGET_CLASSES = [2]  # Cars
QUEUE_SIZE = 32
YOLO_EVERY_N = 5
MIN_CONF = 0.5

# ---------------- LÍNEAS DE CONTEO ----------------
LINE_UP = 320    # Línea de conteo para coches que van arriba
LINE_DOWN = 400  # Línea de conteo para coches que van abajo
OFFSET = 5       # Tolerancia en pixeles para cruzar la línea
EXCLUSION_ZONE = (0, 0, 400, 180)

# ---------------- FUNCIONES ----------------
def in_exclusion(bbox, zone):
    x, y, w, h = bbox
    zx1, zy1, zx2, zy2 = zone
    cx, cy = x + w // 2, y + h // 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

def IoU(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0

def dist_centers(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    c1, c2 = (x1 + w1 // 2, y1 + h1 // 2), (x2 + w2 // 2, y2 + h2 // 2)
    return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

def match_score(b1, b2):
    """Emparejamiento robusto para no perder IDs"""
    iou = IoU(b1, b2)
    d = dist_centers(b1, b2)
    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    area_diff = abs(area1 - area2) / (area1 + area2 + 1e-6)
    score = iou - 0.002 * d - 0.5 * area_diff
    return score

def suprimir_detecciones_solapadas(detecciones, iou_thresh=0.6):
    """Elimina detecciones duplicadas dentro de otra caja"""
    if len(detecciones) <= 1:
        return detecciones
    keep = []
    detecciones = sorted(detecciones, key=lambda b: b[2]*b[3], reverse=True)
    for b1 in detecciones:
        suprimir = False
        for b2 in keep:
            iou = IoU(b1, b2)
            if iou > iou_thresh or (
                b1[0] > b2[0] and b1[1] > b2[1] and 
                b1[0] + b1[2] < b2[0] + b2[2] and 
                b1[1] + b1[3] < b2[1] + b2[3]
            ):
                suprimir = True
                break
        if not suprimir:
            keep.append(b1)
    return keep

# ---------------- INICIALIZACIÓN ----------------
print("Cargando modelo YOLOv8n...")
model = YOLO("yolov8n.pt")
model.overrides['conf'] = 0.25
model.overrides['max_det'] = 50

fvs = FileVideoStream(VIDEO_PATH, queueSize=QUEUE_SIZE).start()
time.sleep(1.5)

trackers = {}
tracker_id = 0
frame_count = 0
count_up = count_down = 0
counted_ids = set()

# ---------------- LOOP PRINCIPAL ----------------
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
    detecciones = []

    # YOLO cada N frames
    if frame_count % YOLO_EVERY_N == 0:
        results = model(frame, verbose=False, device="cpu")
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                if int(cls[i]) in TARGET_CLASSES and conf[i] > MIN_CONF:
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    if not in_exclusion(bbox, EXCLUSION_ZONE):
                        detecciones.append(bbox)
            # Suprimir duplicados dentro del mismo coche
            detecciones = suprimir_detecciones_solapadas(detecciones, 0.6)

    # Actualizar trackers existentes
    nuevos_trackers = {}
    for tid, info in list(trackers.items()):
        ok, bbox = info['tracker'].update(frame)
        if not ok:
            continue
        x, y, w, h = map(int, bbox)
        bbox = (x, y, w, h)

        # Eliminar trackers fuera del frame
        if y + h < 0 or y > frame.shape[0] or x + w < 0 or x > frame.shape[1]:
            continue

        # Mejorar con detecciones cercanas
        mejor_det = None
        mejor_score = -1.0
        for det in detecciones:
            s = match_score(bbox, det)
            if s > mejor_score:
                mejor_score = s
                mejor_det = det
        if mejor_score > 0.2:
            bbox = mejor_det

        info['bbox'] = bbox
        nuevos_trackers[tid] = info

        # Dibujar tracker
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Calcular centro
        cy = y + h // 2
        if 'last_cy' not in info:
            info['last_cy'] = cy

        # Conteo al cruzar líneas
        if tid not in counted_ids:
            # Dirección DOWN
            if info['last_cy'] < LINE_DOWN <= cy:
                count_down += 1
                counted_ids.add(tid)
                print(f"↓ Car {tid} contado como DOWN")
            # Dirección UP
            elif info['last_cy'] > LINE_UP >= cy:
                count_up += 1
                counted_ids.add(tid)
                print(f"↑ Car {tid} contado como UP")
        info['last_cy'] = cy

    trackers = nuevos_trackers

    # Añadir nuevos trackers sin duplicar
    for det in detecciones:
        nuevo = True
        cx1, cy1 = det[0] + det[2]//2, det[1] + det[3]//2

        for tid, info in trackers.items():
            cx2, cy2 = info['bbox'][0] + info['bbox'][2]//2, info['bbox'][1] + info['bbox'][3]//2
            if IoU(det, info['bbox']) > 0.4 or np.hypot(cx1 - cx2, cy1 - cy2) < 50:
                nuevo = False
                break

        if nuevo:
            tracker = cv2.legacy.TrackerMOSSE_create()
            tracker.init(frame, det)
            trackers[tracker_id] = {
                'tracker': tracker,
                'bbox': det,
                'last_cy': det[1] + det[3] // 2
            }
            tracker_id += 1

    # Dibujar líneas
    overlay = frame.copy()
    cv2.line(overlay, (0, LINE_UP), (WIDTH, LINE_UP), (0, 0, 255), 2)
    cv2.line(overlay, (0, LINE_DOWN), (WIDTH, LINE_DOWN), (255, 0, 0), 2)
    cv2.rectangle(overlay, EXCLUSION_ZONE[:2], EXCLUSION_ZONE[2:], (128, 128, 128), -1)
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    cv2.putText(frame, f"UP: {count_up}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"DOWN: {count_down}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Trackers: {len(trackers)}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("YOLO + MOSSE Tracker - Lines Safe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fvs.stop()
cv2.destroyAllWindows()

print("\n=== RESULTADOS FINALES ===")
print(f"UP (abajo→arriba): {count_up}")
print(f"DOWN (arriba→abajo): {count_down}")
print(f"TOTAL: {count_up + count_down}")
print("===========================")
