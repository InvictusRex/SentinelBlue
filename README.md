# SentinelBlue

**Edge-Optimized Deep Learning for Autonomous Maritime Search and Rescue**

---

## Model Selection

SentinelBlue focuses on real-time RGB-based object detection for maritime search-and-rescue scenarios under strict edge deployment constraints. The following models are evaluated to study accuracy–efficiency trade-offs.

### Primary Models

- **YOLOv8n**
  - Baseline model for dataset validation and initial experimentation
  - Fast convergence and stable training behavior
  - Representative of realistic edge-deployment constraints

- **YOLOv11n**
  - Primary model for final experiments
  - Improved feature extraction and small-object detection
  - Main candidate for onboard UAV deployment

- **YOLOv26**
  - Edge-optimized experimental variant
  - Evaluated for scaling behavior under compute and latency constraints

### Comparative Model

- **RT-DETR (Light Variant)**
  - Transformer-based object detector
  - Used for architectural and accuracy comparison
  - Not intended for edge deployment

All models are trained and evaluated using identical datasets and splits to ensure fair comparison.

---

## Object Classes

The final object taxonomy used across all models is fixed and SAR-oriented:

- person
- boat
- jetski
- buoy
- emergency_appliance

### Class Definitions

- **person**: Any human detected in water or on floating platforms
- **boat**: Small to medium watercraft
- **jetski**: Personal watercraft
- **buoy**: Floating buoy or lifebuoy-like objects
- **emergency_appliance**: Rescue and flotation equipment such as life jackets, life rafts, life rings, and throwable rescue devices

Rare rescue equipment is grouped under `emergency_appliance` to improve learnability while preserving SAR semantics.

---

## Dataset Curation Strategy

### Base Dataset

- **SeaDronesSee** is used as the anchor dataset due to its UAV-based maritime imagery and SAR-relevant annotations.

### Supplementary Data

To correct severe class imbalance, external datasets are used **only to augment the training split**, with explicit class remapping and no validation or test contamination.

Supplementary datasets are used to reinforce:

- `jetski`
- `buoy`
- `emergency_appliance`

All dataset modifications are non-destructive and fully reproducible.

### Synthetic Data

- Synthetic data generation is not relied upon as a primary balancing mechanism
- The dataset is instead strengthened using real-world annotated imagery
- This avoids domain drift and preserves visual realism

---

## Training Strategy

- Identical train/validation/test splits across all models
- No class-specific tuning or dataset manipulation per model
- Primary focus on:
  - Recall for the `person` class
  - Per-class Average Precision
  - mAP50 and mAP50–95

---

## Edge Optimization

For onboard UAV inference, trained models are evaluated and optimized using:

- FP16 / INT8 quantization
- Layer fusion and pruning
- TensorRT acceleration
- Benchmarking for:
  - FPS
  - Latency
  - Power consumption
  - Thermal stability

---

## Operational Logic

SentinelBlue follows a **search-and-report** paradigm:

- UAV performs grid-based search over incident regions
- Objects are detected onboard in real time
- Spatial relationships between:
  - persons
  - boats
  - buoys
  - emergency appliances  
    are evaluated
- Detections and metadata are transmitted to a command center
- Final decision-making is performed by a human operator

Autonomous rescue actions and pose-based distress classification are intentionally excluded.

---

## Scope

- RGB-only perception (current)
- RGB–thermal fusion (planned extension)
- Fully autonomous rescue actions are out of scope

---

## Documentation

Detailed design and implementation decisions are documented in:

- [`Documentation/Project_Overview.md`](Documentation/Project%20Overview.md)
- [`Documentation/Model_Selection Strategy.md`](Documentation/Model%20Selection%20Strategy.md)
- [`Documentation/Class_Taxonomy.md`](Documentation/Class%20Taxonomy.md)
- [`Documentation/Dataset_Curation.md`](Documentation/Data%20Curation%20Strategy.md)

Additional documentation will be added as training, evaluation, and deployment stages are completed.

---
