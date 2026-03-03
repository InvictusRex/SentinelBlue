# Training Strategy

## Overview

This document formalizes the training strategy adopted for SentinelBlue, with a specific focus on continued training (checkpoint-based fine-tuning) of YOLO-based object detection models on the finalized SentinelBlue dataset. The training phase is treated as a controlled and isolated stage in the pipeline, where no further dataset modifications are introduced and all performance improvements are attributed strictly to model learning dynamics.

The objective of this stage is not to re-learn generic visual features, but to refine already learned maritime representations and improve detection robustness across SAR-relevant object classes. Particular emphasis is placed on maintaining stability in minority-class learning while preserving high recall for the safety-critical `person` class. The training configuration is therefore designed to balance convergence efficiency, computational feasibility, and experimental rigor.

---

## Training Paradigm

SentinelBlue adopts a continued training paradigm, where training resumes from a previously trained checkpoint rather than starting from randomly initialized weights. The checkpoint used represents the most recent state of prior training on the SentinelBlue dataset, incorporating both learned feature representations and class-specific detection behavior.

This approach is intentionally chosen to avoid the inefficiencies associated with full retraining. Maritime SAR imagery presents highly domain-specific challenges such as water reflections, wave-induced noise, and small object visibility, which require sustained exposure for effective feature learning. Restarting training from scratch would discard these learned representations and significantly increase convergence time.

In addition, continued training provides smoother optimization dynamics, particularly for minority classes that have only recently achieved sufficient representation through augmentation. Reinitializing the model could destabilize these classes and reintroduce imbalance effects at the gradient level. By continuing from an existing checkpoint, the model is allowed to incrementally refine its understanding without losing previously acquired knowledge.

---

## Model Initialization

The model is initialized using a previously saved checkpoint (`last.pt`), loaded through the Ultralytics YOLO framework. This checkpoint contains both backbone and detection head weights, ensuring that the model retains its full representational capacity.

No layers are frozen during initialization. This is a deliberate design decision, as freezing parts of the network would restrict the model’s ability to adapt to the finalized dataset distribution, which differs significantly from generic pretraining datasets. By allowing all layers to update, the model can simultaneously refine low-level feature extraction and high-level object discrimination.

This configuration ensures that previously learned representations are preserved while still enabling meaningful adaptation to SentinelBlue’s SAR-specific data characteristics.

---

## Dataset Integration

Training is performed on the fully finalized SentinelBlue dataset, referenced through a `data.yaml` configuration file. The dataset strictly adheres to the frozen class taxonomy and split structure established during the curation phase.

No augmentations, class remapping, or additional data injections are performed during training. The training split contains all previously augmented samples, while validation and test splits remain completely untouched.

This separation ensures that evaluation metrics reflect true generalization performance. It also guarantees that improvements observed during training arise purely from optimization and model learning, rather than from any changes in dataset composition.

---

## Training Configuration

The training configuration is designed to balance detection performance with computational feasibility under Kaggle constraints.

The model is trained for an additional 50 epochs, which is sufficient for convergence refinement without introducing significant overfitting risk. Image resolution is fixed at 640 pixels, providing an effective trade-off between small-object detectability and memory efficiency.

A batch size of 192 is used across two GPUs. This large batch size reduces gradient variance and stabilizes optimization, which is particularly important for minority classes such as `jetski`, `buoy`, and `emergency_appliance`.

Data loading is parallelized using multiple worker threads, ensuring that GPU utilization remains high and that training is not limited by I/O bottlenecks.

---

## Multi-GPU Strategy

Training is executed on a dual-GPU setup (T4×2) using data parallelism. Batches are distributed across both GPUs, enabling larger effective batch sizes without exceeding individual GPU memory limits.

This configuration significantly reduces training time while maintaining stable gradient updates. It also aligns with Kaggle’s hardware environment, ensuring that experiments remain reproducible and consistent within platform constraints.

The use of multi-GPU training reflects practical deep learning workflows, where models are trained on moderate compute resources before being deployed to edge devices.

---

## Design Rationale

Several constraints were enforced during training to preserve experimental integrity.

The dataset remains completely frozen throughout this phase, ensuring that no data leakage or distribution shifts occur. This allows for fair comparison across different model architectures and training runs.

The dominant `person` class is intentionally retained without downsampling. This reflects real-world SAR conditions and ensures high recall for the most critical detection target. Imbalance is addressed through training dynamics rather than artificial dataset manipulation.

Continued training is preferred over full retraining to preserve learned maritime features and reduce computational overhead, while still enabling the model to adapt to the finalized dataset distribution.

---

## Expected Outcomes

This training configuration is expected to produce stable improvements in detection accuracy across all classes. Minority-class performance is expected to benefit from both prior augmentation and stable gradient updates enabled by large-batch training.

The model should demonstrate strong generalization on validation data, consistent detection of small objects, and high recall for persons under varied maritime conditions.

---

## Final Position

The SentinelBlue training strategy represents a controlled, dataset-consistent, and computationally efficient approach to model refinement. By leveraging checkpoint-based continuation, multi-GPU training, and a strictly frozen dataset, the system ensures that performance improvements are both meaningful and reproducible.
