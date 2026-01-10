# Project Overview

## Motivation

Maritime Search and Rescue (SAR) operations are time-critical, high-risk missions where early visual detection of humans and rescue-relevant objects can directly impact survival outcomes. Unmanned Aerial Vehicles (UAVs) are increasingly used in SAR due to their ability to rapidly cover large oceanic regions, operate in hazardous conditions, and provide real-time situational awareness.

However, effective deployment of UAVs in maritime SAR remains limited by the reliability of onboard perception systems. Challenges such as sea clutter, glare, waves, small object sizes, and extreme class imbalance make maritime visual detection significantly more difficult than terrestrial scenarios.

**SentinelBlue** addresses this challenge by focusing on robust, real-time object detection for maritime SAR using deep learning, with explicit consideration for edge deployment constraints.

---

## Project Objective

The primary objective of SentinelBlue is to design and evaluate a **deep learning–centric perception pipeline** capable of detecting SAR-critical objects from UAV-mounted RGB cameras in real time.

The project aims to:

- detect humans and rescue-relevant objects in maritime environments,
- operate under realistic edge-compute constraints,
- remain robust to severe dataset imbalance,
- and provide reliable perception outputs suitable for downstream SAR decision-making.

Rather than attempting full autonomy, SentinelBlue deliberately isolates and strengthens the **perception layer**, which is the most failure-prone and safety-critical component in SAR pipelines.

---

## Scope Definition

SentinelBlue is explicitly scoped to avoid overreach and maintain technical rigor.

### In Scope

- RGB-based object detection from aerial imagery
- UAV-centric maritime viewpoints
- Edge-deployable deep learning models
- Dataset-centric optimization and class balancing
- Comparative evaluation of lightweight CNN and transformer detectors

---

## Design Philosophy

SentinelBlue follows a **dataset-first engineering philosophy**, recognizing that model performance in safety-critical applications is often bounded more by data quality than architectural sophistication.

Key design principles include:

- **Dataset Integrity over Model Complexity**  
  Emphasis is placed on careful dataset curation, explicit class remapping, and avoidance of label noise.

- **Instance-Level Reasoning**  
  All imbalance analysis and rebalancing decisions are made using object instance counts rather than image counts.

- **Edge Realism**  
  Models are selected and evaluated with practical onboard deployment constraints in mind.

- **Transparency and Reproducibility**  
  All dataset modifications are performed using deterministic, non-destructive scripts and documented explicitly.

---

## System Perspective

From a systems viewpoint, SentinelBlue operates as a **search-and-report perception module** within a larger SAR framework.

At a high level:

- A UAV performs a predefined search pattern over a maritime region.
- Onboard deep learning models detect SAR-relevant objects in real time.
- Detected objects and metadata are transmitted to a ground control station.
- Final situational assessment and rescue decisions remain human-driven.

This human-in-the-loop approach aligns with current operational SAR practices and regulatory expectations.

---
