
#FUNCIONA AMB SEQLONG1
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
YOLO_EVERY_N = 3
MIN_CONF = 0.3

LINE_REF = 500
EXCLUSION_ZONE = (0, 0, 400, 180)

# Paràmetres de moviment
REVERSAL_FRAMES = 300  # frames per considerar reversió del mateix cotxe
MOVE_THRESHOLD = 25    # desplaçament mínim per comptar
STATIC_HISTORY = 20  # historial per detectar estàtic
STATIC_DIST = 5    # distància màxima per considerar estàtic


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


def is_static(positions, dist_thresh=STATIC_DIST, history=STATIC_HISTORY):

    if len(positions) < history:
        return False
    pts = positions[-history:]
    dists = [np.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]) for i in range(1, len(pts))]
    return sum(dists) < dist_thresh


# ---------------- PROGRAMA PRINCIPAL ----------------
def main():
    print("Cargando modelo YOLOv8n...")
    model = YOLO("yolov8n.pt")
    model.overrides['conf'] = MIN_CONF
    model.overrides['max_det'] = 50

    ct = CentroidTracker(maxDisappeared=40)
    fvs = FileVideoStream(VIDEO_PATH, queueSize=QUEUE_SIZE).start()
    time.sleep(0.2)

    frame_count = 0
    count_up = 0
    count_down = 0

    object_last_y = {}
    object_movement = {}
    object_initial_y = {}
    object_last_event = {}  # {objectID: ("UP"/"DOWN", frame_number)}
    last_detections = []

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

        # --- Dibuixar deteccions YOLO ---
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "car", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Actualitzar tracker ---
        objects = ct.update(rects, frame)

        for (objectID, centroid) in objects.items():
            cX, cY = centroid

            # Guardar historial de moviment
            object_movement.setdefault(objectID, []).append((cX, cY))
            if len(object_movement[objectID]) > 30:
                object_movement[objectID] = object_movement[objectID][-30:]

            estacionado = is_static(object_movement[objectID])
            object_initial_y.setdefault(objectID, cY)
            desplazamiento_total = abs(cY - object_initial_y[objectID])

            # --- Comprovar travessament línia amb reversió ---
            if not estacionado and desplazamiento_total > MOVE_THRESHOLD:
                if objectID in object_last_y:
                    prevY = object_last_y[objectID]

                    # ↓↓ travessa cap avall ↓↓
                    if prevY < LINE_REF <= cY:
                        if objectID in object_last_event:
                            last_dir, last_frame = object_last_event[objectID]
                            if last_dir == "UP" and (frame_count - last_frame) < REVERSAL_FRAMES:
                                count_up = max(0, count_up - 1)
                                print(f"[Reversal] ID {objectID} ha revertit (UP→DOWN)")
                                object_last_event.pop(objectID, None)
                            else:
                                count_down += 1
                                object_last_event[objectID] = ("DOWN", frame_count)
                        else:
                            count_down += 1
                            object_last_event[objectID] = ("DOWN", frame_count)

                    # ↑↑ travessa cap amunt ↑↑
                    elif prevY > LINE_REF >= cY:
                        if objectID in object_last_event:
                            last_dir, last_frame = object_last_event[objectID]
                            if last_dir == "DOWN" and (frame_count - last_frame) < REVERSAL_FRAMES:
                                count_down = max(0, count_down - 1)
                                print(f"[Reversal] ID {objectID} ha revertit (DOWN→UP)")
                                object_last_event.pop(objectID, None)
                            else:
                                count_up += 1
                                object_last_event[objectID] = ("UP", frame_count)
                        else:
                            count_up += 1
                            object_last_event[objectID] = ("UP", frame_count)

            object_last_y[objectID] = cY

            # --- Dibuixar ID i estat ---
            color = (255, 0, 0) if not estacionado else (100, 100, 100)
            estado = "MOVING" if not estacionado else "STATIC"
            cv2.circle(frame, (cX, cY), 4, color, -1)
            cv2.putText(frame, f"ID:{objectID} {estado}", (cX - 10, cY - 10),
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

        cv2.imshow("YOLO + Centroid Tracker (reversal + static filter)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Finalitzar ---
    fvs.stop()
    cv2.destroyAllWindows()

    print("\n=== RESULTATS FINALS ===")
    print(f"UP (abajo→arriba): {count_up}")
    print(f"DOWN (arriba→abajo): {count_down}")
    print(f"TOTAL: {count_up + count_down}")
    print("===========================")


if __name__ == "__main__":
    main()


