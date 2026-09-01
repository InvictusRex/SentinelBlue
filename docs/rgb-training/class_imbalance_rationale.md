# Class Balance and Dataset Size Rationale

## Overview

After controlled augmentation, the SentinelBlue training split contains the following object instance distribution:

| Class               | Train Instances |
| ------------------- | --------------: |
| person              |          52,200 |
| boat                |          18,109 |
| jetski              |           9,722 |
| buoy                |           9,705 |
| emergency_appliance |           8,837 |

At first glance, the `person` class appears significantly larger than the remaining classes. This document explains why the dataset was intentionally **not downsampled**, and why this class distribution is both technically sound and operationally realistic for maritime Search and Rescue (SAR).

---

## 1. SAR Is Inherently Person-Centric

SentinelBlue is designed for maritime Search and Rescue (SAR). In real-world SAR scenarios:

- Humans in distress are the primary detection target.
- Boats, buoys, and emergency appliances provide contextual information.
- Missing a person has significantly higher operational cost than over-detecting equipment.

Therefore, it is expected and desirable that the `person` class remains dominant in the dataset. Artificially reducing `person` samples would distort the real-world distribution and weaken recall performance for the most safety-critical class.

---

## 2. Minority Classes Are No Longer Underrepresented

Before augmentation, rare classes (e.g., `jetski`, `buoy`, `emergency_appliance`) suffered from instance starvation. After controlled augmentation:

- All non-person classes now exceed ~8,000 instances.
- Each class has sufficient representation for stable gradient learning.
- No class remains in the low-data regime (<2,000 instances).

Modern object detectors such as YOLOv8 and YOLOv11 are robust when each class has several thousand instances. The current distribution does not constitute harmful imbalance.

---

## 3. YOLO Does Not Learn Class Priors Like a Classifier

Unlike traditional image classifiers, YOLO-based object detectors:

- First predict objectness,
- Then classify objects conditionally.

The model does not simply learn global class frequency priors. As long as each class appears in sufficient quantity, larger classes do not automatically suppress smaller ones.

With ~9,000 instances for each non-person class, SentinelBlue does not fall into the regime where class collapse is likely.

---

## 4. Downsampling Would Harm Generalization

Downsampling the `person` class was deliberately avoided for the following reasons:

1. Loss of environmental diversity  
   Person instances occur under varying:
   - lighting conditions,
   - sea states,
   - camera altitudes,
   - background clutter.

   Removing samples would reduce environmental robustness.

2. Contextual co-occurrence  
   Persons often appear alongside:
   - boats,
   - buoys,
   - emergency appliances.

   Downsampling could disrupt realistic spatial relationships.

3. Increased overfitting risk  
   Fewer person examples may lead to overfitting specific poses or backgrounds.

Given that SAR prioritizes person recall, reducing person data would be counterproductive.

---

## 5. Dataset Size Is Not Excessive

The final dataset contains approximately:

- 124,352 total object instances
- ~98,000 training instances

For context:

- COCO contains ~860k training instances.
- VisDrone contains ~540k instances.
- DOTA contains over 2 million instances.

SentinelBlue remains a medium-scale detection dataset and is fully manageable within modern GPU training constraints (e.g., Kaggle T4 GPUs).

Training time is primarily dependent on the number of images rather than raw object instance counts. Therefore, the current dataset size is computationally practical.

---

## 6. Preferred Solution: Training-Time Adjustments

Rather than altering dataset realism, imbalance concerns are addressed during training through:

- Early stopping
- Mosaic scheduling
- Potential class-weight tuning
- Per-class recall monitoring
- Confusion matrix analysis

This approach preserves dataset integrity while maintaining analytical rigor.

---

## Final Position

The SentinelBlue dataset was intentionally preserved without downsampling the `person` class because:

- The class distribution reflects realistic SAR scenarios.
- All other classes have achieved sufficient representation.
- Downsampling would reduce generalization capability.
- Modern detection models can handle this distribution effectively.

The dataset is therefore considered balanced, defensible, and ready for training without further structural modification.
