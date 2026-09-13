import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from threading import Thread
from queue import Queue
from ultralytics import YOLO

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.q = Queue(maxsize=3)
        self.stopped = False
        self.thread = Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                break
            if not self.q.full():
                self.q.put(frame)
            else:
                try:
                    self.q.get_nowait()
                    self.q.put(frame)
                except:
                    pass

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def release(self):
        self.stopped = True
        self.capture.release()


class ActiveLearningHarvest:
    """JD Focus: Active learning loops, edge-case data capture & quality curation"""
    def __init__(self, save_dir="data/hard_samples", conf_range=(0.30, 0.55)):
        self.save_dir = save_dir
        self.low_conf, self.high_conf = conf_range
        self.last_saved = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def trigger_harvest(self, frame, conf):
        # CONFUSION ZONE: agar confidence 30% - 55% ke beech ho toh auto-harvest karo
        current_time = time.time()
        if self.low_conf <= conf <= self.high_conf and (current_time - self.last_saved > 3.0):
            filename = os.path.join(self.save_dir, f"sample_{int(current_time)}.jpg")
            cv2.imwrite(filename, frame)
            self.last_saved = current_time


class SafetyRulesEngine:
    """JD Focus: Spatio-temporal CEP, Dwell-Time noise rejection, JSON Metadata logging"""
    def __init__(self, danger_zone_poly, dwell_threshold_seconds=2.0, log_path="data/violations.json"):
        self.danger_zone = np.array(danger_zone_poly, dtype=np.int32).reshape((-1, 1, 2))
        self.dwell_threshold = dwell_threshold_seconds
        self.log_path = log_path
        self.entry_times = {}
        self.logged_alarms = set()

    def is_inside(self, foot_point):
        pt = (float(foot_point[0]), float(foot_point[1]))
        return cv2.pointPolygonTest(self.danger_zone, pt, False) >= 0

    def log_violation(self, track_id, duration):
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": int(track_id),
            "event_type": "PROLONGED_RESTRICTED_ZONE_BREACH",
            "dwell_duration_sec": round(duration, 2)
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def evaluate(self, track_id, foot_point):
        inside = self.is_inside(foot_point)
        current_time = time.time()
        alert = False

        if inside:
            if track_id not in self.entry_times:
                self.entry_times[track_id] = current_time
            elapsed = current_time - self.entry_times[track_id]
            if elapsed >= self.dwell_threshold:
                alert = True
                if track_id not in self.logged_alarms:
                    self.log_violation(track_id, elapsed)
                    self.logged_alarms.add(track_id)
        else:
            if track_id in self.entry_times:
                del self.entry_times[track_id]
            if track_id in self.logged_alarms:
                self.logged_alarms.remove(track_id)

        return inside, alert


def run_pipeline(model_path="yolov8n.pt", video_src=0):
    print(f"[INFO] Initializing Enterprise Pipeline: {model_path}...")
    model = YOLO(model_path)
    cam = ThreadedCamera(video_src)
    active_harvester = ActiveLearningHarvest()
    time.sleep(1.0)

    danger_poly = [(150, 150), (450, 150), (500, 420), (100, 420)]
    engine = SafetyRulesEngine(danger_poly, dwell_threshold_seconds=2.0)

    prev_time = time.time()
    fps = 0.0

    print("[INFO] Pipeline started. Press 'q' to stop.")
    while True:
        frame = cam.read()
        if frame is None:
            continue

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else fps
        prev_time = curr_time

        results = model.track(frame, persist=True, classes=[0], verbose=False)

        cv2.polylines(frame, [engine.danger_zone], True, (0, 255, 255), 2)
        cv2.putText(frame, "RESTRICTED DANGER ZONE", (160, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        violation = False
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = box
                foot_point = (int((x1 + x2) / 2), int(y2))
                inside, alert = engine.evaluate(track_id, foot_point)

                # Active Learning: edge cases harvest
                active_harvester.trigger_harvest(frame, conf)

                color = (0, 255, 0)
                label = f"ID: {track_id} ({conf:.2f})"
                if inside:
                    color = (0, 165, 255)
                    label += " [BREACH]"
                if alert:
                    color = (0, 0, 255)
                    label += " [CRITICAL ALARM]"
                    violation = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, foot_point, 5, (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if violation:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(frame, "CRITICAL SAFETY VIOLATION LOGGED", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("EdgeSight-CV Production Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()
