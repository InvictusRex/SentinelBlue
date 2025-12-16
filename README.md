# SentinelBlue

Edge-Optimized Deep Learning for Autonomous Maritime Search and Rescue

---

## Model Selection

SentinelBlue focuses on real-time RGB-based object detection for maritime search-and-rescue scenarios under edge deployment constraints. The following models are used:

### Primary Models

- **YOLOv8n**

  - Used as the baseline model
  - Fast iteration, dataset validation, and early experimentation
  - Strong edge deployment support

- **YOLOv11n**
  - Main model for final experiments
  - Better feature extraction and small-object detection
  - Primary candidate for onboard UAV deployment

### Comparative Model

- **RT-DETR (Light Variant)**
  - Transformer-based detector
  - Used for accuracy and architectural comparison
  - Not intended for edge deployment

All models are trained and evaluated using identical datasets and splits to ensure fair comparison.

---

## Object Classes

The final class taxonomy used across all models is:

- person
- boat
- jetski
- buoy
- emergency_appliance

### Class Definitions

- **person**: Any human detected in water or on floating platforms
- **boat**: Small to medium watercraft
- **jetski**: Personal watercraft
- **buoy**: Floating buoy or lifebuoy
- **emergency_appliance**: Rescue or flotation equipment (life jackets, rafts, rescue floats, etc.)

This class design prioritizes SAR relevance and minimizes class imbalance by grouping rare rescue equipment.

---

## Dataset Curation Strategy

### Base Dataset

- **SeaDronesSee** is used as the anchor dataset due to its UAV-based maritime imagery and SAR-relevant annotations.

### Supplementary Data

- Additional images are sourced only for underrepresented classes such as:
  - `buoy`
  - `emergency_appliance`

Supplementary data is restricted to maritime environments to preserve distribution consistency.

### Synthetic Data

- Synthetic data generation is applied selectively for rare classes
- Techniques include object compositing onto water backgrounds with scale, rotation, blur, and lighting variation
- Synthetic data is capped to avoid domain drift

---

## Training Strategy

- Identical train/validation/test splits across all models
- Focus on high recall for the `person` class
- Monitoring:
  - Recall
  - False negatives
  - Per-class AP
  - mAP50 / mAP50–95

---

## Edge Optimization

For onboard inference, trained models undergo:

- FP16 / INT8 quantization
- Channel pruning and layer fusion
- TensorRT acceleration
- Benchmarking for:
  - FPS
  - Latency
  - Power consumption
  - Thermal stability

---

## Operational Logic

SentinelBlue follows a **search-and-report** paradigm:

- UAV performs grid search over incident coordinates
- Objects are detected onboard in real time
- Spatial relationships between:
  - persons
  - boats
  - buoys
  - emergency appliances
    are evaluated
- Detections and metadata are reported to a command center
- Final validation is performed by a human operator

Pose estimation and autonomous distress classification are intentionally excluded at this stage.

---

## Scope

- RGB-only detection (current)
- RGB–thermal fusion (planned extension)
- Fully autonomous rescue actions are out of scope

---
