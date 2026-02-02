# Data Augmentation Methodology

## Overview

The SentinelBlue dataset underwent controlled, class-conditional augmentation to address severe instance imbalance in safety-critical object categories.

Augmentation was applied **exclusively to the training split**, while validation and test splits were preserved without modification to maintain evaluation integrity.

The objective was not artificial balancing, but rather to ensure that all classes achieved sufficient representation for stable and robust deep learning training.

---

## Initial Class Imbalance

Prior to augmentation, the dataset exhibited the following imbalance:

- `person` was heavily dominant.
- `boat` was reasonably represented.
- `jetski`, `buoy`, and `emergency_appliance` were underrepresented.
- `emergency_appliance` was critically sparse.

Imbalance analysis was conducted using **object instance counts**, not image counts, since instance frequency directly impacts gradient stability in object detection models.

---

## Augmentation Policy

The following strict policies were enforced:

1. **Training split only**
   - No augmentation was applied to validation or test sets.
   - No synthetic data was introduced into evaluation splits.

2. **Class-conditional augmentation**
   - Only underrepresented classes were augmented.
   - Dominant classes (e.g., `person`) were not artificially reduced or oversampled.

3. **Controlled growth**
   - Each class was augmented to a target range (~8k–10k instances).
   - No class was fully equalized.
   - Realistic SAR distribution was preserved.

4. **Non-destructive workflow**
   - Original images were never modified or deleted.
   - Augmented samples were added with new filenames.
   - Reproducible scripts were used for all transformations.

---

## Augmentation Strategies by Class

### 1. Jetski

Jetskis are small, fast-moving watercraft that are often visually confused with boats.

Augmentation techniques:

- Random scale jitter
- Limited rotation (±10°)
- Motion blur (light)
- Brightness and contrast variation
- Gaussian noise

Rationale:

- Simulates altitude changes and UAV motion.
- Encourages discrimination from visually similar boats.
- Improves robustness to glare and water reflections.

---

### 2. Buoy

Buoys are small floating objects with low contrast against dynamic water backgrounds.

Augmentation techniques:

- Stronger scale jitter (small-object bias)
- Brightness and contrast variation
- Mild blur
- Gaussian noise

Rationale:

- Improves small-object detection performance.
- Encourages learning under varied sea-state conditions.
- Enhances robustness to motion and environmental noise.

---

### 3. Emergency Appliance (Copy-Paste Augmentation)

The `emergency_appliance` class includes:

- Life jackets
- Life rings
- Life rafts
- Flotation devices

This class is safety-critical and visually diverse.

A hybrid augmentation strategy was used:

#### A. Copy-Paste Augmentation (Primary Method)

- Emergency appliance objects were cropped using bounding boxes.
- Crops were pasted into new training images at physically plausible locations.
- Bounding boxes were recalculated and appended to YOLO labels.
- Placement ensured objects remained within image bounds.

Rationale:

- Generates new contextual object occurrences.
- Teaches spatial relationships between persons and rescue equipment.
- Increases rare-object frequency without synthetic rendering.

Copy-paste augmentation is widely used in modern object detection pipelines and is academically defensible when restricted to training data.

#### B. Post-Paste Photometric Augmentation

- Brightness and contrast adjustment
- Mild blur
- Gaussian noise

Rationale:

- Ensures pasted objects integrate visually into maritime scenes.
- Improves robustness to environmental lighting variations.

---

## Augmentations Explicitly Avoided

The following transformations were intentionally excluded:

- Horizontal and vertical flips (physically implausible maritime context)
- Heavy rotation (violates upright object assumptions)
- Aggressive hue shifts (color is discriminative for SAR objects)
- Synthetic texture generation
- GAN-based synthesis

These exclusions ensured preservation of semantic realism.

---

## Final Dataset Distribution

After augmentation:

- All non-person classes exceed ~8,000 training instances.
- The `person` class remains dominant, reflecting real SAR conditions.
- Validation and test splits remain unchanged.

The final dataset contains approximately 124,000 total object instances across all splits.

---

## Justification for Not Downsampling the Person Class

The `person` class was intentionally not downsampled because:

- SAR operations are inherently person-centric.
- Downsampling would reduce environmental diversity.
- Non-person classes achieved sufficient representation post-augmentation.
- YOLO-based detectors do not collapse solely due to frequency imbalance when minority classes are adequately represented.

Class balance was addressed by strengthening minority classes rather than weakening the primary class.

---

## Dataset Freeze

Following augmentation:

- No further dataset modifications were performed.
- No additional synthetic data was introduced.
- All future improvements are confined to training configuration and model selection.

The dataset is considered finalized and ready for systematic experimentation.
