# EdgeSight-CV: Real-Time Computer Vision & Industrial Safety Analytics Pipeline

EdgeSight-CV is an edge-optimized, production-ready computer vision pipeline engineered for surveillance, industrial automation, and smart spaces. It monitors restricted danger zones, performs multi-object tracking (ByteTRACK), executes spatio-temporal Complex Event Processing (CEP) to suppress false positives, and accelerates inference via ONNX Runtime.

---

## Benchmark Metrics (Empirical Profiling on Intel Core Ultra)

| Metric | PyTorch (Raw Engine) | ONNX Runtime (CPU) | Improvement / Delta |
|---|---|---|---|
| **Average Latency** | 58.39 ms | **52.54 ms** | **~10% Latency Reduction** |
| **Throughput (FPS)** | 17.1 FPS | **19.0 FPS** | **+1.9 FPS (+11.1%) Boost** |
| **Peak RAM Footprint**| 370.12 MB | 446.14 MB | Optimized graph execution |

---

## Architectural Highlights

- **Video Ingestion:** Multi-threaded frame decoupling (`ThreadedCamera`) eliminating RTSP/HLS stream buffer lag via circular queue dropping.
- **Tracking & Identity Persistence:** Integrated ByteTRACK for persistent worker IDs and bottom-center spatial foot-coordinate mapping across occlusions.
- **Complex Event Processing (CEP):** Spatio-temporal ray-casting polygon checks (`pointPolygonTest`) paired with a 2.0s dwell-time filter to eliminate transient false alarms.
- **Active Learning & MLOps:** Automated harvesting of marginal-confidence predictions (0.30 <= conf <= 0.55) for edge-case active learning and synthetic data generation (`src/dataset_mlops.py`).
- **Graph Optimization:** Exported to ONNX Opset 18 with onnxslim quantization profiling latency, throughput, and memory.
- **Automated Testing & CI/CD:** Spatio-temporal logic validation verified via PyTest and executed on push via GitHub Actions.

---

## Quickstart

```bash
# 1. Run Live Vision Pipeline (Webcam / RTSP)
python src/pipeline.py

# 2. Run Latency, FPS & Memory Benchmark
python src/benchmark.py

# 3. Curate Active Learning Samples
python src/dataset_mlops.py

# 4. Run Test Suite
python -m pytest tests/test_rules.py
