# 👁️ Retinal Optic Nerve Head Segmentation

> Connected Component Labeling (CCL) based segmentation of the Optic Disc and Optic Cup in retinal fundus images — for automated Glaucoma screening via Cup-to-Disc Ratio (CDR) computation.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Overview

Glaucoma is a chronic eye disease that causes irreversible blindness through optic nerve damage. Its primary diagnostic indicator is the **Cup-to-Disc Ratio (CDR)** — the ratio of the optic cup size to the optic disc size. An enlarged CDR signals glaucomatous damage.

This project implements an automated segmentation pipeline for the **Optic Nerve Head (ONH)** using **Connected Component Labeling (CCL)** with 8-connectivity, without relying on deep learning. The pipeline produces 3-class pixel-wise labels:

| Label | Region |
|-------|--------|
| 0 | Background (retina & corners) |
| 1 | Optic Disc (outer complete disc) |
| 2 | Optic Cup (inner brightest core) |

---

## 🔬 Pipeline

```
[Retinal Fundus Image]
        │
        ▼
① Pre-processing
   Color channel selection
   Thresholding using V set (train-derived)
        │
        ▼
② Connected Component Analysis — 8-connectivity
   Separate Optic Disc from background
   → 2 labels: Background | Optic Disc
        │
        ▼
③ OD Refinement
   Apply new V set inside OD region
   Differentiate Optic Cup from Optic Disc
   → 3 labels: Background | Optic Disc | Optic Cup
        │
        ▼
④ Post-processing
   Morphological cleanup
   Largest component selection
        │
        ▼
⑤ Evaluation — Dice Coefficient
   Per-class: Background, Optic Disc, Optic Cup
   True pixels  = overlap with ground truth mask
   False pixels = non-overlap
   Normalized by total true pixels in ground truth
        │
        ▼
[Segmented Output + CDR + Dice Scores]
```

---

## ✨ Key Implementation Details

### V Set Design
The pixel inclusion set V was designed empirically from training images — analyzing intensity distributions in the green channel (highest contrast for ONH structures) to determine optimal thresholds for separating:
- OD from background (high brightness region)
- Optic Cup from OD (brightest inner core)

### 8-Connectivity CCL
Pixels are connected if they share an edge or corner AND their value belongs to V. The largest connected component after labeling is selected as the target structure.

### Dice Coefficient
```
Dice = (2 × |Pred ∩ GT|) / (|Pred| + |GT|)
```
Computed per class against provided ground truth masks from the test set.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **OpenCV** — image I/O, color conversion, morphological ops
- **NumPy** — array operations, CCL implementation
- **Matplotlib** — result visualization

---

## 📁 Project Structure

```
retinal-segmentation/
│
├── segmentation.py          # Main pipeline
├── ccl.py                   # 8-connectivity CCL implementation
├── vset.py                  # V set definition and thresholding
├── dice.py                  # Dice coefficient computation
├── visualize.py             # Result visualization and comparison
│
├── dataset/
│   ├── train/               # Training images (for V set design)
│   │   ├── images/
│   │   └── masks/
│   └── test/                # Test images (for Dice evaluation)
│       ├── images/
│       └── masks/
│
├── outputs/                 # Segmented results
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
pip install numpy opencv-python matplotlib
```

```bash
# Run segmentation on test set + compute Dice scores
python segmentation.py --data dataset/test/

# Visualize results for a single image
python visualize.py --image dataset/test/images/sample.jpg
```

---

## 📊 Results

> *(Add your Dice coefficient scores and result images here)*

| Class | Dice Coefficient |
|-------|-----------------|
| Background | - |
| Optic Disc | - |
| Optic Cup | - |

**Sample Results:**

| Original | Ground Truth (OD) | Ground Truth (Cup) | Segmented Output |
|----------|------------------|-------------------|-----------------|
| *(img)* | *(img)* | *(img)* | *(img)* |

---

## 💡 Key Learnings

- **Why green channel?** — The green channel provides the highest contrast between the bright ONH and surrounding retinal tissue in fundus images.
- **Why 8-connectivity?** — The optic disc is a continuous blob — 4-connectivity risks splitting it at diagonal boundaries. 8-connectivity keeps it whole.
- **Why largest component?** — CCL may produce multiple foreground components from noise. The optic disc is always the largest foreground region.
- **CDR significance** — CDR > 0.6 is generally flagged as suspicious for Glaucoma in clinical screening.

---

## 📚 Dataset

Retinal fundus images with pixel-wise annotated ground truth masks for Optic Disc and Optic Cup.  
*(DRISHTI-inspired dataset provided as part of EC312 Digital Image Processing coursework, NUST)*

---

## 👤 Author

**Abu-Bakar Chaudhary**  
Computer Engineering · NUST · Class of 2027  
EC312 Digital Image Processing  
[GitHub](https://github.com/abu-bakarchaudhary) · [LinkedIn](https://linkedin.com/in/abubakar-chaudhary-ce45)
