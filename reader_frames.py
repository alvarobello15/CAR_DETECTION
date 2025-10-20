import cv2
from threading import Thread
from queue import Queue
import time

class FileVideoStream:
    def __init__(self, path, queueSize=128):
        print(f"[DEBUG] Inicializando FileVideoStream con video: {path}")
        self.stream = cv2.VideoCapture(path)
        
        # Verificar que el video se abrió correctamente
        if not self.stream.isOpened():
            print(f"[ERROR] No se pudo abrir el video: {path}")
            return
        
        print(f"[DEBUG] Video abierto exitosamente")
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        print("[DEBUG] Iniciando thread de lectura")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        print("[DEBUG] Thread iniciado")
        return self
        
    def update(self):
        print("[DEBUG] Update thread started - leyendo frames")
        frame_count = 0
        
        while True:
            if self.stopped:
                print(f"[DEBUG] Thread detenido. Frames leídos: {frame_count}")
                return
            
            # Intentamos leer un frame
            (grabbed, frame) = self.stream.read()
            frame_count += 1
            
            if not grabbed:
                print(f"[DEBUG] Fin del video. Frames totales: {frame_count}")
                self.stop() 
                return
            
            # Si la cola está llena, esperamos un poco
            if self.Q.full():
                time.sleep(0.01)
            else:
                self.Q.put(frame)
                if frame_count % 30 == 0:  # Imprimir cada 30 frames
                    print(f"[DEBUG] {frame_count} frames leídos. Queue size: {self.Q.qsize()}")
                
    def read(self):
        return self.Q.get()
        
    def more(self):
        return self.Q.qsize() > 0
        
    def stop(self):
        print("[DEBUG] Parando FileVideoStream")
        self.stopped = True


# Test rápido
if __name__ == "__main__":
    print("Iniciando test de FileVideoStream...")
    fvs = FileVideoStream("output7.mp4", queueSize=8).start()
    
    print("Esperando 2 segundos para que se llene la cola...")
    time.sleep(2)
    
    print(f"Queue size: {fvs.Q.qsize()}")
    print(f"fvs.more(): {fvs.more()}")
    
    if fvs.more():
        frame = fvs.read()
        print(f"Frame leído: {frame.shape}")
    
    fvs.stop()
    time.sleep(0.5)
    print("Test finalizado")