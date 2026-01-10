## Object Classes and Taxonomy

SentinelBlue uses a **fixed, unified object taxonomy** designed specifically for **maritime Search and Rescue (SAR)** perception from UAV platforms.

The class design prioritizes:

- SAR relevance over fine-grained categorization,
- visual detectability in RGB imagery,
- and robustness against class imbalance.

The taxonomy is intentionally compact to ensure reliable learning under real-world constraints.

---

### Final Class Schema (Frozen)

The following five classes are used consistently across all datasets, models, and experiments:

| Class ID | Class Name          |
| -------: | ------------------- |
|        0 | person              |
|        1 | boat                |
|        2 | jetski              |
|        3 | buoy                |
|        4 | emergency_appliance |

This schema is **frozen** and is not modified during training or evaluation.

---

### Class Definitions and SAR Semantics

#### **person**

Any human visible in the maritime environment, including:

- people in water,
- people on vessels,
- partially submerged individuals.

This is the **primary SAR target**, and recall for this class is treated as mission-critical.

---

#### **boat**

Small to medium watercraft, including:

- fishing boats,
- recreational boats,
- rigid inflatable boats (RIBs).

Large ships are not explicitly targeted, as SAR operations typically focus on smaller vessels in distress scenarios.

---

#### **jetski**

Personal watercraft, treated as a distinct class due to:

- unique visual characteristics,
- high speed and maneuverability,
- frequent presence in near-shore rescue and accident scenarios.

Jetskis are separated from boats to reduce visual confusion and improve detection specificity.

---

#### **buoy**

Floating buoy-like objects, including:

- navigation buoys,
- lifebuoys when not directly worn,
- marker buoys.

Buoys are important contextual cues in SAR operations, often indicating:

- navigation hazards,
- man-overboard markers,
- or proximity to rescue infrastructure.

---

#### **emergency_appliance**

Rescue and survival equipment deployed during emergencies, including:

- life jackets / personal flotation devices (PFDs),
- life rafts,
- life rings,
- throwable flotation devices,
- other life-saving appliances.

This class groups visually diverse but **functionally identical SAR objects** to:

- reduce extreme class sparsity,
- improve learnability,
- and preserve semantic relevance.

Fine-grained categorization of emergency equipment is intentionally avoided.

---

### Design Rationale

- **Functional grouping over object specificity**  
  Objects are grouped based on their role in SAR, not manufacturing differences.

- **Avoidance of rare micro-classes**  
  Extremely sparse categories were merged to prevent unstable training.

- **Consistency across datasets**  
  All external datasets are remapped to this taxonomy prior to training.

This taxonomy reflects a **practical SAR perception system**, not a generic object detector.

---
