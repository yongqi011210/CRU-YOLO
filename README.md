# CRU-YOLO: Collaborative Refinement for Rotated Object Detection

## 🔍 Key Features

- **Cascade Rotated Refinement (Cascade-YOLO)**
  - Multi-stage coarse-to-fine refinement of rotated bounding boxes
  - Joint optimization of object center, scale, and orientation
  - Improves localization stability and reduces regression noise in complex scenes

- **Cross-Stage Attention (CSA)**
  - Explicit information interaction between consecutive cascade stages
  - Alleviates error accumulation during multi-stage refinement
  - Enhances consistency across iterative regression stages

- **Rotation-Consistent Feature Alignment (RCFA)**
  - Inserted before multi-scale feature fusion to align orientation-sensitive features
  - Reduces cross-scale rotational inconsistency in feature representations
  - Strengthens rotation-consistent semantic features for downstream detection

- **Rotation-Aware Bounding Box Regression (RABR)**
  - Direction-enhanced regression branch for stabilized OBB prediction
  - Supports multiple variants: RABR-S, RABR-C, RABR-M, and RABR-MC
  - RABR-MC is recommended as the default configuration in our experiments

- **Ultralytics-Compatible Design**
  - Fully compatible with the Ultralytics YOLO-OBB training and inference pipeline
  - All components are enabled directly via YAML configuration
  - No modification to the overall training workflow is required

## 📁 Repository Structure
```text
CRU-YOLO/
├── ultralytics/
│   └── nn/
│       └── modules/
│           ├── obb_cascade.py      # Cascade rotated OBB head (Cascade-YOLO) with CSA and RABR
│           ├── rcfa.py             # Rotation-Consistent Feature Alignment (RCFA)
│           └── __init__.py
├── configs/
│   └── cascade-yolov8-obb.yaml     # CRU-YOLO model configuration
├── train.py
├── README.md
└── requirements.txt
```

## ⚙️ Installation

```bash
git clone https://github.com/yongqi011210/CRU-YOLO.git
cd CRU-YOLO
pip install -r requirements.txt
```

- Python ≥ 3.8
- PyTorch ≥ 1.13
- Ultralytics YOLO with OBB support

## 🚀 Usage

### Register custom modules

```python
# ultralytics/nn/tasks.py
from ultralytics.nn.modules.obb_cascade import OBB_CascadeHead
from ultralytics.nn.modules.rcfa import RCFA
```

```python
# ultralytics/nn/modules/__init__.py
from .obb_cascade import OBB_CascadeHead
from .rcfa import RCFA
```

### Model configuration (YAML)

```yaml
- [[18, 21, 24], 1, OBB_CascadeHead, [nc, 1, 2, True, "mc", False, False]]
```

```text
[nc, ne, cascade_stages, use_csa, rabr_mode, return_all_stages, debug]
```
- cascade_stages: number of cascade refinement stages

- use_csa: enable Cross-Stage Attention between stages

- rabr_mode: select RABR variant from {s, c, m, mc}

- return_all_stages: output intermediate predictions for analysis

- debug: enable debug logging


### Training

```bash
yolo obb train model=cascade-yolov8-obb.yaml data=ssdd.yaml imgsz=512 batch=4 epochs=50 device=0 pretrained=False workers=0 name=yolov8-obb
```


## 📊 Notes on Training and Stride Inference

Ultralytics performs a dummy forward pass during model construction to automatically infer feature strides.

To maintain compatibility with this mechanism, CRU-YOLO ensures that the cascade rotated detection head returns valid tensor outputs during stride inference.
The full multi-stage refinement process is activated only during training and inference, thereby enabling cascade optimization without affecting stride estimation.

## 🧪 Supported Applications

- SAR ship detection in near-shore and off-shore scenes
- Remote sensing rotated object detection
- Arbitrary-oriented object detection in complex backgrounds


## 📌 Citation

```bibtex
@article{CRUYOLO,
  title   = {CRU-YOLO: Collaborative Refinement for Rotated Object Detection},
  author  = {Kang, Yongqi},
  journal = {Under Review},
  year    = {2026}
}
```

```bibtex
@software{ultralytics_yolo,
  author  = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  title   = {Ultralytics YOLO},
  year    = {2023},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```


## 📜 License

This project is released under the **Apache License 2.0**.

CRU-YOLO is built upon **Ultralytics YOLO**, which is licensed under **AGPL-3.0**. Users must comply with the terms of both licenses.


## 🙌 Acknowledgements

- Ultralytics YOLO
- OpenMMLab
- Public SAR ship detection datasets (RSDD-SAR, SSDD+)
