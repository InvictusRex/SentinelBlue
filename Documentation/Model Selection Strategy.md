## Model Selection

SentinelBlue evaluates multiple object detection architectures to study the trade-offs between **accuracy, robustness, and edge deployability** in maritime SAR scenarios.
All models are trained and evaluated on the **same frozen SentinelBlue dataset**, using identical splits and evaluation metrics, to ensure a fair and controlled comparison.

---

### Primary Detection Models

#### **YOLOv8n** (Baseline)

YOLOv8n serves as the **baseline model** for the project.

- Lightweight, single-stage detector
- Fast convergence and stable training behavior
- Well-suited for early dataset validation and debugging
- Representative of realistic edge-deployment constraints

YOLOv8n is used to:

- validate dataset quality after curation,
- establish reference performance metrics,
- and provide a lower-bound benchmark for comparison.

---

#### **YOLOv11n** (Primary Experimental Model)

YOLOv11n is the **primary model** used for final experiments and analysis.

- Improved feature extraction compared to YOLOv8n
- Better handling of small and visually ambiguous objects
- More expressive backbone while remaining edge-friendly
- Suitable for deployment on embedded UAV platforms

YOLOv11n represents the **best balance** between accuracy and efficiency within the YOLO family for this application.

---

### Additional Comparative Models

#### **YOLOv26** (Edge-Optimized Variant)

YOLOv26 is included as an **edge-optimized experimental model**.

- Designed to improve efficiency under strict compute constraints
- Evaluated for latency, throughput, and power-aware performance
- Used to study scaling behavior as model capacity increases

This model helps analyze how architectural complexity impacts SAR-relevant detection under edge limitations.

---

#### **RT-DETR** (Lightweight Variant)

RT-DETR is included as a **transformer-based baseline** for architectural comparison.

- End-to-end transformer detector
- Eliminates hand-crafted anchor design
- Typically offers strong global context modeling

RT-DETR is **not intended for edge deployment**, but is used to:

- compare CNN-based detectors against transformer-based approaches,
- study performance differences on cluttered maritime scenes,
- and contextualize YOLO-family results.

---

### Model Evaluation Policy

- No model-specific dataset tuning
- No class-specific architectural bias
- Identical training and evaluation protocols across models

This ensures that observed performance differences arise from **model architecture**, not dataset manipulation.

---
