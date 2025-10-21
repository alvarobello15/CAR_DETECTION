# yolo_csrt_counter.py - ULTRA ROBUSTO
import cv2
import imutils
import time
from ultralytics import YOLO
from reader_frames import FileVideoStream
import numpy as np

# ------------------- CONFIG -------------------
VIDEO_PATH = "output7.mp4"
TARGET_CLASSES = [2]
QUEUE_SIZE = 32
YOLO_EVERY_N = 2  # YOLO cada 2 frames para detección rápida
WIDTH = 400  # Mayor resolución para mejor detección

# Zonas más amplias para capturar mejor
ZONE_UP = (0, 200, 400, 350)
ZONE_DOWN = (0, 500, 400, 650)

# ------------------- FUNCIONES -------------------
def in_zone(bbox, zone):
    """Verifica si el centro del bbox está en la zona"""
    x, y, w, h = bbox
    zx1, zy1, zx2, zy2 = zone
    cx = x + w // 2
    cy = y + h // 2
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

def IoU(bbox1, bbox2):
    """Calcula IoU entre dos bboxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    
    return inter / union if union > 0 else 0

def distancia_centros(bbox1, bbox2):
    """Calcula distancia entre centros de dos bboxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    cx1, cy1 = x1 + w1//2, y1 + h1//2
    cx2, cy2 = x2 + w2//2, y2 + h2//2
    return ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5

# ------------------- INICIALIZACIÓN -------------------
model = YOLO("yolov8n.pt")
fvs = FileVideoStream(VIDEO_PATH, queueSize=QUEUE_SIZE).start()
time.sleep(1.5)

trackers_info = {}
tracker_id_counter = 0
frame_count = 0
count_up = 0
count_down = 0
counted_ids = set()  # IDs ya contados para evitar duplicados

MAX_FRAMES_SIN_DETECCION = 15  # Mantener tracker más tiempo
MIN_CONFIANZA = 0.25  # Confianza mínima YOLO

# ------------------- LOOP PRINCIPAL -------------------
while True:
    if fvs.more():
        frame = fvs.read()
        if frame is None:
            continue
            
        frame = imutils.resize(frame, width=WIDTH)
        frame_count += 1

        # Ejecutar YOLO con más frecuencia
        run_yolo = (frame_count % YOLO_EVERY_N == 1) or len(trackers_info) == 0
        detecciones_yolo = []
        
        if run_yolo:
            results = model(frame, conf=MIN_CONFIANZA, verbose=False, device='cpu', iou=0.5)
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for i in range(len(xyxy)):
                        if int(cls_ids[i]) in TARGET_CLASSES:
                            x1, y1, x2, y2 = map(int, xyxy[i])
                            bbox = (x1, y1, x2-x1, y2-y1)
                            conf = float(confs[i])
                            detecciones_yolo.append((bbox, conf))

        # Actualizar trackers existentes
        lost_trackers = []
        trackers_bboxes = {}
        
        for t_id, info in list(trackers_info.items()):
            success, bbox = info['tracker'].update(frame)
            
            if success:
                x, y, w, h = map(int, bbox)
                bbox = (x, y, w, h)
                trackers_bboxes[t_id] = bbox
                info['frames_sin_deteccion'] = 0
                
                # Reinicializar tracker si hay detección cercana (mejora precisión)
                if run_yolo:
                    for det_bbox, det_conf in detecciones_yolo:
                        if IoU(bbox, det_bbox) > 0.3 or distancia_centros(bbox, det_bbox) < 50:
                            # Reiniciar con detección de YOLO
                            new_tracker = cv2.legacy.TrackerCSRT_create()
                            new_tracker.init(frame, det_bbox)
                            info['tracker'] = new_tracker
                            bbox = det_bbox
                            trackers_bboxes[t_id] = bbox
                            break
                
                x, y, w, h = bbox
                
                # Color según estado
                color = (0, 255, 0)
                if info['zones']['up']:
                    color = (255, 165, 0)  # Naranja si pasó por UP
                if info['zones']['down']:
                    color = (0, 165, 255)  # Azul si pasó por DOWN
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"ID:{t_id}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Actualizar zonas
                prev_up = info['zones']['up']
                prev_down = info['zones']['down']
                
                if in_zone(bbox, ZONE_UP):
                    info['zones']['up'] = True
                if in_zone(bbox, ZONE_DOWN):
                    info['zones']['down'] = True

                # Contar solo si no ha sido contado antes
                if t_id not in counted_ids:
                    # UP->DOWN = count_down
                    if info['zones']['up'] and info['zones']['down']:
                        if prev_up and not prev_down:
                            count_down += 1
                            counted_ids.add(t_id)
                            lost_trackers.append(t_id)
                            print(f"✓ Car {t_id} counted DOWN (total: {count_down})")
                        elif not prev_up and prev_down:
                            count_up += 1
                            counted_ids.add(t_id)
                            lost_trackers.append(t_id)
                            print(f"✓ Car {t_id} counted UP (total: {count_up})")
            else:
                # Tracker falló, incrementar contador
                info['frames_sin_deteccion'] += 1
                if info['frames_sin_deteccion'] > MAX_FRAMES_SIN_DETECCION:
                    lost_trackers.append(t_id)

        # Eliminar trackers perdidos
        for t_id in lost_trackers:
            if t_id in trackers_info:
                del trackers_info[t_id]

        # Añadir nuevos trackers
        if run_yolo:
            for det_bbox, det_conf in detecciones_yolo:
                # Verificar que no se solape con trackers existentes
                es_nuevo = True
                for t_id, existing_bbox in trackers_bboxes.items():
                    if IoU(det_bbox, existing_bbox) > 0.3:
                        es_nuevo = False
                        break
                    if distancia_centros(det_bbox, existing_bbox) < 40:
                        es_nuevo = False
                        break
                
                if es_nuevo:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    success = tracker.init(frame, det_bbox)
                    if success:
                        trackers_info[tracker_id_counter] = {
                            'tracker': tracker,
                            'zones': {'up': False, 'down': False},
                            'frames_sin_deteccion': 0
                        }
                        print(f"+ New tracker ID:{tracker_id_counter}")
                        tracker_id_counter += 1

        # Dibujar zonas con mejor visualización
        overlay = frame.copy()
        cv2.rectangle(overlay, ZONE_UP[:2], ZONE_UP[2:], (0, 0, 255), -1)
        cv2.rectangle(overlay, ZONE_DOWN[:2], ZONE_DOWN[2:], (255, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

        # Bordes de zonas
        cv2.rectangle(frame, ZONE_UP[:2], ZONE_UP[2:], (0, 0, 255), 3)
        cv2.putText(frame, "ZONA UP", (ZONE_UP[0]+10, ZONE_UP[1]+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.rectangle(frame, ZONE_DOWN[:2], ZONE_DOWN[2:], (255, 0, 0), 3)
        cv2.putText(frame, "ZONA DOWN", (ZONE_DOWN[0]+10, ZONE_DOWN[1]+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Información en pantalla
        cv2.rectangle(frame, (0, 0), (200, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"UP: {count_up}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"DOWN: {count_down}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Trackers: {len(trackers_info)}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("YOLO + CSRT Robust Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        if fvs.stopped:
            break

fvs.stop()
cv2.destroyAllWindows()
print(f"\n{'='*40}")
print(f"CONTEO FINAL:")
print(f"  UP (abajo→arriba): {count_up}")
print(f"  DOWN (arriba→abajo): {count_down}")
print(f"  TOTAL: {count_up + count_down}")
print(f"{'='*40}")
