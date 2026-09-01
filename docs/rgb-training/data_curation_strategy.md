## Dataset Curation Strategy

Dataset quality and class balance are treated as **first-class design constraints** in SentinelBlue.  
Given the absence of manual data collection, the project adopts a **dataset-centric deep learning approach**, where careful curation and controlled augmentation are prioritized over architectural complexity.

---

### Base Dataset: SeaDronesSee

The SentinelBlue dataset is anchored on the **SeaDronesSee** UAV maritime object detection dataset.

SeaDronesSee is selected because it provides:

- real-world UAV imagery over maritime environments,
- SAR-relevant object annotations,
- realistic camera viewpoints and altitudes,
- and challenging visual conditions such as glare, waves, and clutter.

The original train/validation/test splits from SeaDronesSee are preserved as the **structural foundation** of SentinelBlue.

---

### Class Imbalance Diagnosis

Initial inspection revealed severe **object-instance imbalance**, particularly for safety-critical classes:

- `person` was overwhelmingly dominant,
- `boat` was reasonably represented,
- `jetski`, `buoy`, and especially `emergency_appliance` were severely underrepresented.

Importantly, all imbalance analysis was performed using **instance counts rather than image counts**, as object frequency directly affects learning stability in detection models.

---

### External Dataset Augmentation (TRAIN Only)

To address imbalance while preserving evaluation integrity:

- External datasets were used **exclusively to augment the training split**.
- Validation and test sets were **never modified or polluted**.
- All external annotations were **explicitly remapped** to the frozen SentinelBlue class schema.

The following datasets were used for targeted rebalancing:

- **SeaDronesSee (Base Dataset)**  
  https://universe.roboflow.com/ntnu-2wibj/seadronessee-odv2

- **Jet Ski Detection Dataset**  
  https://universe.roboflow.com/mouda-deguerre-cr65w/jet-ski-detection

- **Buoy Detection Dataset**  
  https://universe.roboflow.com/yolo-project/buoy

- **Life Jacket Detection Dataset**  
  https://universe.roboflow.com/ai-project-zczg5/life-jacket-on

- **Life-Saving Appliances Dataset**  
  https://universe.roboflow.com/microg-mpeko/microg-zipup

External splits (`train`, `val`, `test`) were intentionally ignored, and all usable samples were merged into SentinelBlue’s training set only.

---

### Label Integrity and Noise Avoidance

Several safeguards were enforced during curation:

- No pseudo-labeling was applied after discovering class collapse risks.
- Images with missing or ambiguous annotations were excluded.
- External classes not semantically aligned with maritime SAR (e.g., non-relevant equipment) were explicitly removed.
- Mixed-label images were filtered to retain only valid SAR-relevant objects.

This ensured that performance gains arose from **genuine supervision**, not synthetic or noisy labels.

---

### Final Dataset Balance

After all curation steps:

- All five classes achieved **sufficient representation** for stable training.
- Safety-critical classes (`person`, `emergency_appliance`) received priority.
- Rare-class sparsity was eliminated without oversampling or artificial synthesis.

Once finalized, the dataset was **frozen**, and no further modifications were permitted.

---

### Reproducibility

All dataset transformations were performed using:

- deterministic Python scripts,
- explicit class-ID remapping,
- and non-destructive copy-based workflows.

This guarantees full reproducibility and auditability of the SentinelBlue dataset.

---
