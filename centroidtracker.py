from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict

"""
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects
    
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, maxDisappeared=50, use_color=True):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bboxes = OrderedDict()  # guardar bounding boxes actuals
        self.maxDisappeared = maxDisappeared
        self.use_color = use_color
        self.color_hist = {}  # histogrames HSV per objecte

    # ---------------- FUNCIONS AUXILIARS ----------------
    def IoU(self, b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    def dist_centers(self, b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        c1, c2 = (x1 + w1 // 2, y1 + h1 // 2), (x2 + w2 // 2, y2 + h2 // 2)
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def get_color_hist(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def color_similarity(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return 0
        sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, min(1, sim))

    def match_score(self, b1, b2, frame=None, hist1=None, hist2=None):
        iou = self.IoU(b1, b2)
        d = self.dist_centers(b1, b2)
        area1 = b1[2] * b1[3]
        area2 = b2[2] * b2[3]
        area_diff = abs(area1 - area2) / (area1 + area2 + 1e-6)

        # Color opcional
        color_sim = 0
        if self.use_color:
            if hist1 is None:
                hist1 = self.get_color_hist(frame, b1)
            if hist2 is None:
                hist2 = self.get_color_hist(frame, b2)
            color_sim = self.color_similarity(hist1, hist2)

        # Score ponderat
        score = iou - 0.002 * d - 0.5 * area_diff + 0.3 * color_sim
        return score

    # ---------------- TRACKER PRINCIPAL ----------------
    def register(self, centroid, bbox, frame=None):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        if self.use_color and frame is not None:
            self.color_hist[self.nextObjectID] = self.get_color_hist(frame, bbox)
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]
        if objectID in self.color_hist:
            del self.color_hist[objectID]

    def update(self, rects, frame=None):
        if len(rects) == 0:
            # Si no hi ha deteccions, incrementa desapareguts
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)

        # Si no hi ha objectes registrats
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                x1, y1, x2, y2 = rects[i]
                bbox = (x1, y1, x2 - x1, y2 - y1)
                self.register(inputCentroids[i], bbox, frame)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Creem matriu de similitud amb heurístiques
            scores = np.zeros((len(objectCentroids), len(inputCentroids)), dtype=float)
            for i, objectID in enumerate(objectIDs):
                for j, rect in enumerate(rects):
                    x1, y1, x2, y2 = rect
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    hist1 = self.color_hist.get(objectID) if self.use_color else None
                    scores[i, j] = self.match_score(self.bboxes[objectID], bbox, frame, hist1)

            # Triar màxima similitud (no mínima distància)
            rows = scores.max(axis=1).argsort()[::-1]
            cols = scores.argmax(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if scores[row, col] < 0.1:  # llindar per descartar falsos match
                    continue

                objectID = objectIDs[row]
                x1, y1, x2, y2 = rects[col]
                bbox = (x1, y1, x2 - x1, y2 - y1)
                self.objects[objectID] = inputCentroids[col]
                self.bboxes[objectID] = bbox
                self.disappeared[objectID] = 0
                if self.use_color and frame is not None:
                    self.color_hist[objectID] = self.get_color_hist(frame, bbox)
                usedRows.add(row)
                usedCols.add(col)

            # Objectes no assignats → incrementa desapareguts
            unusedRows = set(range(scores.shape[0])).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Noves deteccions no assignades → registra
            unusedCols = set(range(scores.shape[1])).difference(usedCols)
            for col in unusedCols:
                x1, y1, x2, y2 = rects[col]
                bbox = (x1, y1, x2 - x1, y2 - y1)
                self.register(inputCentroids[col], bbox, frame)

        return self.objects
