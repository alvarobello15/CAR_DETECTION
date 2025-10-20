"""# yolo_csrt_counter.py
import cv2
import imutils
import time
from ultralytics import YOLO
from reader_frames import FileVideoStream
import numpy as np

# ------------------- CONFIG -------------------
VIDEO_PATH = "output7.mp4"
TARGET_CLASSES = [2]  # 2 = coche en COCO
QUEUE_SIZE = 8
YOLO_RUN_EVERY_N_FRAMES = 1  # Ejecutar YOLO cada N frames

# Definir zonas de conteo (x1, y1, x2, y2)
ZONE_UP = (0,350, 370, 450)      # zona superior
ZONE_DOWN = (0, 550, 370, 700)   # zona inferior

# ------------------- FUNCIONES -------------------
def in_zone(bbox, zone):
    """Devuelve True si el centro del bbox está dentro de la zona"""
    x, y, w, h = bbox
    zx1, zy1, zx2, zy2 = zone
    cx = x + w // 2
    cy = y + h // 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

def get_histogram(frame, bbox):
    """Calcula histograma HSV normalizado del parche del coche"""
    x, y, w, h = bbox
    patch = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [16,16], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(hist1, hist2):
    """Compara histogramas usando correlación"""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# ------------------- INICIALIZACIÓN -------------------
model = YOLO("yolov8m.pt")
fvs = FileVideoStream(VIDEO_PATH, queueSize=QUEUE_SIZE).start()
time.sleep(2)

trackers_info = {}  # t_id -> {'tracker': TrackerCSRT, 'zones': {'up':False,'down':False}, 'hist':histograma}
tracker_id_counter = 0
frame_count = 0
count_up = 0
count_down = 0

# ------------------- LOOP PRINCIPAL -------------------
while True:
    if fvs.more():
        frame = fvs.read()
        frame = imutils.resize(frame, width=400)
        frame_count += 1

        # Ejecutar YOLO cada N frames o si no hay trackers
        if len(trackers_info) == 0 or frame_count % YOLO_RUN_EVERY_N_FRAMES == 0:
            results = model(frame,conf=0.35)
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for i in range(len(xyxy)):
                        cls_id = int(cls_ids[i])
                        if cls_id in TARGET_CLASSES:
                            x1, y1, x2, y2 = map(int, xyxy[i])
                            bbox = (x1, y1, x2-x1, y2-y1)
                            hist = get_histogram(frame, bbox)
                            conf = confs[i]  # confianza de YOLO


                            # Evitar duplicados: comparar con histogramas existentes
                            duplicate = False
                            for info in trackers_info.values():
                                if compare_hist(hist, info['hist']) > 0.7:  # umbral
                                    duplicate = True
                                    break
                            if duplicate:
                                continue

                            # Crear tracker CSRT
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(frame, bbox)
                            trackers_info[tracker_id_counter] = {
                                'tracker': tracker,
                                'zones': {'up': False, 'down': False},
                                'hist': hist,
                                'conf': conf   
                            }

                            tracker_id_counter += 1

        # Actualizar trackers
        lost_trackers = []
        for t_id, info in trackers_info.items():
            tracker = info['tracker']
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Car {t_id} {info['conf']:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


                # Actualizar zonas
                if in_zone(bbox, ZONE_UP):
                    info['zones']['up'] = True
                if in_zone(bbox, ZONE_DOWN):
                    info['zones']['down'] = True

                # Contar coches
                if info['zones']['up'] and info['zones']['down']:
                    count_down += 1
                    lost_trackers.append(t_id)
                elif info['zones']['down'] and info['zones']['up']:
                    count_up += 1
                    lost_trackers.append(t_id)
            else:
                lost_trackers.append(t_id)

        # Eliminar trackers perdidos o que completaron conteo
        for t_id in lost_trackers:
            del trackers_info[t_id]

        # Dibujar zonas en rojo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, ZONE_UP[:2], ZONE_UP[2:], (0,0,255), -1)    # relleno rojo
        cv2.rectangle(overlay, ZONE_DOWN[:2], ZONE_DOWN[2:], (0,0,255), -1) # relleno rojo
        alpha = 0.2  # transparencia
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Dibujar bordes y textos encima
        cv2.rectangle(frame, ZONE_UP[:2], ZONE_UP[2:], (0,0,255), 2)
        cv2.putText(frame, "UP ZONE", (ZONE_UP[0], ZONE_UP[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.rectangle(frame, ZONE_DOWN[:2], ZONE_DOWN[2:], (0,0,255), 2)
        cv2.putText(frame, "DOWN ZONE", (ZONE_DOWN[0], ZONE_DOWN[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


        # Mostrar contadores
        cv2.putText(frame, f"UP: {count_up}  DOWN: {count_down}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Mostrar frame
        cv2.imshow("YOLO + CSRT Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        if fvs.stopped:
            break

fvs.stop()
cv2.destroyAllWindows()"""

# yolo_csrt_counter_improved.py
import cv2
import imutils
import time
from ultralytics import YOLO
from reader_frames import FileVideoStream
import numpy as np

# ------------------- CONFIG -------------------
VIDEO_PATH = "output7.mp4"
TARGET_CLASSES = [2]  # 2 = coche en COCO
QUEUE_SIZE = 8
YOLO_RUN_EVERY_N_FRAMES = 5  # Ejecutar YOLO cada N frames (puedes poner 1 para todo frame)
CONF_THRESHOLD = 0.35       # confianza mínima para considerar detección
MIN_DET_AREA = 500          # area mínima bbox (en px) para ignorar objetos muy pequeños/ruidos
PENDING_CONFIRMATIONS = 2   # número de runs de YOLO en los que debe aparecer para confirmar
IOU_DUPLICATE_THRESH = 0.5  # IoU > 0.5 => considerar misma detección / evitar duplicado
MAX_MISSED_FRAMES = 15      # eliminar tracker si falla tantos frames seguidos

# Definir zonas de conteo (x1, y1, x2, y2) — ajústalas visualmente
ZONE_UP = (0,350, 370, 450)      # zona superior
ZONE_DOWN = (0, 550, 370, 700)   # zona inferior

# ------------------- UTILIDADES -------------------
def iou(boxA, boxB):
    # boxes en formato (x,y,w,h)
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    a_x2 = ax + aw
    a_y2 = ay + ah
    b_x2 = bx + bw
    b_y2 = by + bh

    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(a_x2, b_x2)
    y_bottom = min(a_y2, b_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = aw * ah
    boxB_area = bw * bh
    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return iou

def in_zone(bbox, zone):
    x, y, w, h = bbox
    zx1, zy1, zx2, zy2 = zone
    cx = x + w // 2
    cy = y + h // 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

def get_histogram(frame, bbox):
    x, y, w, h = bbox
    # proteger recortes fuera de imagen
    h_img, w_img = frame.shape[:2]
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(w_img, x + w); y2 = min(h_img, y + h)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [16,16], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(hist1, hist2):
    if hist1 is None or hist2 is None:
        return -1
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# ------------------- INICIALIZACIÓN -------------------
model = YOLO("yolov8m.pt")
fvs = FileVideoStream(VIDEO_PATH, queueSize=QUEUE_SIZE).start()
time.sleep(2)

trackers_info = {}  # id -> {'tracker', 'zones', 'hist', 'conf', 'last_seen_frame', 'missed'}
tracker_id_counter = 0
pending_detections = []  # lista de {'bbox','hist','count','last_frame','conf'}
frame_count = 0
count_up = 0
count_down = 0

# ------------------- LOOP PRINCIPAL -------------------
while True:
    if fvs.more():
        frame = fvs.read()
        # aumenta resolución si quieres mejor detección (CPU/GPU más uso)
        frame = imutils.resize(frame, width=640)
        frame_count += 1

        # --- Ejecutar YOLO cada N frames ---
        if frame_count % YOLO_RUN_EVERY_N_FRAMES == 0:
            results = model(frame, conf=CONF_THRESHOLD)
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    detections = []
                    for i in range(len(xyxy)):
                        cls_id = int(cls_ids[i])
                        conf = float(confs[i])
                        if cls_id not in TARGET_CLASSES:
                            continue  # solo coches
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        w = x2 - x1; h = y2 - y1
                        if w*h < MIN_DET_AREA:
                            continue  # ignorar detecciones muy pequeñas
                        bbox = (x1, y1, w, h)
                        hist = get_histogram(frame, bbox)
                        detections.append({'bbox': bbox, 'hist': hist, 'conf': conf})

                    # Match detections vs trackers (evitar crear duplicados)
                    for det in detections:
                        bbox_det = det['bbox']
                        # si coincide fuertemente con un tracker existente: saltar
                        is_dup = False
                        for t_id, info in trackers_info.items():
                            last_bbox = info.get('last_bbox', None)
                            if last_bbox is None:
                                continue
                            if iou(bbox_det, last_bbox) > IOU_DUPLICATE_THRESH:
                                is_dup = True
                                break
                        if is_dup:
                            continue

                        # Match con pending_detections: si similar, incrementar count
                        matched = False
                        for p in pending_detections:
                            if iou(bbox_det, p['bbox']) > IOU_DUPLICATE_THRESH or compare_hist(det['hist'], p['hist']) > 0.7:
                                p['bbox'] = bbox_det  # actualizar bbox al más reciente
                                p['hist'] = det['hist'] if det['hist'] is not None else p['hist']
                                p['count'] += 1
                                p['last_frame'] = frame_count
                                p['conf'] = max(p.get('conf', 0.0), det['conf'])
                                matched = True
                                break
                        if not matched:
                            # nueva pending detection
                            pending_detections.append({
                                'bbox': bbox_det,
                                'hist': det['hist'],
                                'count': 1,
                                'last_frame': frame_count,
                                'conf': det['conf']
                            })

                    # Confirmar pending detections si count >= PENDING_CONFIRMATIONS
                    new_pending = []
                    for p in pending_detections:
                        # si ha pasado mucho tiempo desde la última vez, descartar
                        if frame_count - p['last_frame'] > YOLO_RUN_EVERY_N_FRAMES * 4:
                            continue
                        if p['count'] >= PENDING_CONFIRMATIONS:
                            # crear tracker
                            tracker = cv2.legacy.TrackerCSRT_create()
                            x, y, w, h = p['bbox']
                            try:
                                tracker.init(frame, p['bbox'])
                            except Exception:
                                continue
                            trackers_info[tracker_id_counter] = {
                                'tracker': tracker,
                                'zones': {'up': False, 'down': False},
                                'hist': p['hist'],
                                'conf': p['conf'],
                                'last_seen_frame': frame_count,
                                'missed': 0,
                                'last_bbox': p['bbox']
                            }
                            tracker_id_counter += 1
                        else:
                            new_pending.append(p)
                    pending_detections = new_pending

        # --- Actualizar trackers cada frame ---
        lost_trackers = []
        for t_id, info in list(trackers_info.items()):
            tracker = info['tracker']
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                info['last_bbox'] = (x, y, w, h)
                info['last_seen_frame'] = frame_count
                info['missed'] = 0

                # dibujar bbox e id+conf
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                txt = f"Car {t_id} {info.get('conf',0):.2f}"
                cv2.putText(frame, txt, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # zonas
                if in_zone((x,y,w,h), ZONE_UP):
                    info['zones']['up'] = True
                if in_zone((x,y,w,h), ZONE_DOWN):
                    info['zones']['down'] = True

                # contar: si ha pasado por ambas zonas (comprobar orden si es necesario)
                if info['zones']['up'] and info['zones']['down']:
                    count_down += 1
                    lost_trackers.append(t_id)
                elif info['zones']['down'] and info['zones']['up']:
                    count_up += 1
                    lost_trackers.append(t_id)
            else:
                # tracker perdió el objeto en este frame
                info['missed'] += 1
                if info['missed'] > MAX_MISSED_FRAMES:
                    lost_trackers.append(t_id)

        # eliminar trackers perdidos
        for t_id in lost_trackers:
            if t_id in trackers_info:
                del trackers_info[t_id]

        # limpiar pending_detections muy antiguas
        pending_detections = [p for p in pending_detections if frame_count - p['last_frame'] <= YOLO_RUN_EVERY_N_FRAMES * 4]

        # ------------------- DIBUJO ZONAS ROJAS (semi-transparente) -------------------
        overlay = frame.copy()
        cv2.rectangle(overlay, ZONE_UP[:2], ZONE_UP[2:], (0,0,255), -1)
        cv2.rectangle(overlay, ZONE_DOWN[:2], ZONE_DOWN[2:], (0,0,255), -1)
        alpha = 0.18
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.rectangle(frame, ZONE_UP[:2], ZONE_UP[2:], (0,0,255), 2)
        cv2.putText(frame, "UP ZONE", (ZONE_UP[0], ZONE_UP[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.rectangle(frame, ZONE_DOWN[:2], ZONE_DOWN[2:], (0,0,255), 2)
        cv2.putText(frame, "DOWN ZONE", (ZONE_DOWN[0], ZONE_DOWN[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Mostrar contadores y número de trackers/pending
        cv2.putText(frame, f"UP: {count_up}  DOWN: {count_down}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Trackers: {len(trackers_info)}  Pending: {len(pending_detections)}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Mostrar frame
        cv2.imshow("YOLO + CSRT Counter (improved)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        if fvs.stopped:
            break

fvs.stop()
cv2.destroyAllWindows()

