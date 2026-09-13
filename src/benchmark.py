import time
import psutil
import os
import numpy as np
from ultralytics import YOLO

def profile_model(model_name, runs=50):
    print(f"\n================ Profiling: {model_name} ================")
    model = YOLO(model_name)
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = model(dummy, verbose=False)

    latencies = []
    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(dummy, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency

    print(f"Average Latency : {avg_latency:.2f} ms")
    print(f"Throughput      : {fps:.1f} FPS")
    print(f"Peak RAM        : {mem_after:.2f} MB")
    return avg_latency, fps

if __name__ == "__main__":
    profile_model("yolov8n.pt")
    if os.path.exists("yolov8n.onnx"):
        profile_model("yolov8n.onnx")
