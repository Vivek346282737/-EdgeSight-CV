# EdgeSight-CV: Production Industrial Safety & Vision Pipeline

Edge-optimized, low-latency computer vision pipeline engineered for industrial safety zones, human tracking, spatio-temporal dwell time violation detection, and ONNX runtime acceleration.

## Architecture Highlights
- Video Ingestion: Multi-threaded frame decoupling to eliminate RTSP/HLS stream buffer lag.
- Tracking & Re-ID: Integrated ByteTRACK tracking logic with spatial coordinate mapping.
- Complex Event Processing (CEP): Spatial-temporal polygon ray-casting with dwell-time noise filtering.
- Optimization: ONNX Runtime graph acceleration benchmarking latency, throughput (FPS), and RAM footprint.
- Software Engineering: Unit tests for rules validation and modular architecture.

## How to Run
- python src/pipeline.py
- python src/benchmark.py
- python -m unittest tests/test_rules.py
