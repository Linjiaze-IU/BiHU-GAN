# BiHU-GAN  
**HU‑Preserving Bidirectional Cycle GAN for Non‑Contrast and Contrast‑Enhanced CT Synthesis in Head‑and‑Neck Radiotherapy**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> In this repository, Hounsfield Units (HU) refer to the CT number scale used for radiotherapy dose calculation.
>
> Official PyTorch implementation of **BiHU-GAN** – a physically constrained bidirectional cycle‑consistent GAN that simultaneously synthesizes contrast‑enhanced CT from non‑contrast CT and recovers HU‑faithful non‑contrast CT from contrast‑enhanced CT, specifically designed for head‑and‑neck radiotherapy planning.
> 
> We are currently working to extend BiHU‑GAN to more anatomical sites (thorax, abdomen, pelvis, etc.), and sincerely hope to have the honor of continuing to share our research progress and improvements with peers in the field.

---

## 📌 Overview

Radiotherapy planning for head‑and‑neck cancer faces an inherent trade‑off:  
- **Non‑contrast CT (NCCT)** provides accurate CT number for dose calculation but lacks soft‑tissue contrast.  
- **Contrast‑enhanced CT (CECT)** improves target delineation but introduces iodine‑induced CT number shifts, compromising dose accuracy.

**BiHU‑GAN** resolves this by offering bidirectional synthesis with explicit CT number preservation:

| Direction | Clinical Use |
|-----------|---------------|
| **NCCT → CECT** | Generate synthetic contrast‑enhanced CT (sRT‑CECT) to assist contouring without contrast agent injection. |
| **CECT → NCCT** | Remove contrast effects to produce HU‑faithful synthetic non‑contrast CT (sRT‑CT) for accurate dose calculation. |

The model incorporates **HU‑deviation loss**, **gradient‑consistency loss**, and **CBAM attention** to maintain anatomical fidelity, edge sharpness, and physical CT number accuracy – validated on multicenter data with dosimetric endpoints.

---

## ✨ Key Features

- **Bidirectional unpaired translation** – works with paired or unpaired CT volumes.
- **HU‑aware constraints** – tissue‑weighted HU loss (soft‑tissue vs. bone) enforces physical radiodensity fidelity.
- **Gradient consistency loss** – preserves high‑density boundaries (bone, metal‑adjacent regions).
- **CBAM modules** – enhance low‑contrast structures (vessels, lymph nodes).
- **Monte Carlo Dropout** – supports uncertainty estimation during inference.
- **Full DICOM integration** – read/write DICOM series with preserved metadata (RescaleSlope/Intercept, positioning).
- **Built‑in evaluation** – masked SSIM, PSNR, NMAE, HU‑MAD, and contouring/dosimetric metrics.
- **Optimized training** – mixed precision (AMP), gradient accumulation, cosine warm‑up, early stopping.

---

## 🧬 Repository Structure

```
BiHU-GAN/
├── configs/
│   └── bihugan.yaml               # Main configuration file
├── bihugan/
│   ├── config.py                  # YAML config parser
│   ├── data/
│   │   └── volume_paired_dataset.py   # paired volume loading & augmentation
│   ├── models/
│   │   ├── generator.py           # ResNet‑based generator with CBAM
│   │   ├── discriminator.py       # PatchGAN discriminator
│   │   ├── resnet_blocks.py       # Residual blocks
│   │   └── cbam.py                # Channel & spatial attention
│   ├── losses/
│   │   ├── gan_lsgan.py           # LSGAN adversarial loss
│   │   ├── cycle.py               # Cycle and L1 loss
│   │   ├── gradients.py           # Sobel gradient consistency loss
│   │   └── hu_loss.py             # HU Consistency Loss
│   └── utils/
│       ├── metrics.py             # SSIM, PSNR, NMAE, HU‑MAD
│       └── export_dicom.py        # DICOM series export
├── train.py                       # Training entry point
├── test_infer.py                  # Inference & evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/Linjiaze-IU/BiHU-GAN.git
cd BiHU-GAN
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8 / 3.9
- PyTorch ≥ 2.0
- CUDA 11.7 / 11.8 (recommended)

### 2. Data Preparation

You need to prepare **paired or unpaired** CT volumes (NCCT and CECT) from your own institution.  
The dataset loader expects a **text file** listing **all slice paths** of the source modality (e.g., all NCCT DICOM files).  
It automatically derives the corresponding target modality path by replacing a keyword (e.g., `AC0` → `AC1`) – see `pair_from` / `pair_to` in the config.

**Example format** (`train.txt`):
```
/path/to/patient_001/NCCT/slice_001.dcm
/path/to/patient_001/NCCT/slice_002.dcm
...
```

The paired volume is loaded by matching the series directory structure.  
All preprocessing (CT number clipping, normalization, rigid registration, augmentation) is handled inside `PairedVolumeDataset`.

### 3. Configuration

Edit `configs/bihugan.yaml` to set your data paths, resolution, CT number window, loss weights, etc.

**Key parameters**:

| Parameter | Description |
|-----------|-------------|
| `train_list` / `val_list` / `test_list` | Text files with slice paths |
| `pair_from` / `pair_to` | Keyword replacement for modality pairing (e.g., `"AC0"` → `"AC1"`) |
| `hu_min`, `hu_max` | CT number clipping range (default `-1000`, `2047`) |
| `body_hu_thresh` | Threshold for body mask (default `-600`) |
| `bone_hu_thresh` | Threshold separating soft tissue and bone (default `250`) |
| `global_scaling_factor` | Global multiplier for non‑HU losses (recommended `1.2`) |
| `hu_loss_weight` | Weight for HU consistency loss (recommended `1.2`) |
| `model_input_resolution` | Training resolution (`1024` recommended for head‑neck) |
| `epochs`, `batchSize`, `lr` | Training hyperparameters |

> **Note**: The model was trained on 1024×1024 axial slices. For external 512×512 data, the loader automatically upsamples during inference.

### 4. Training

```bash
python train.py --config configs/bihugan.yaml
```

- Checkpoints are saved in `checkpoints_dir/` (both `*_best.pth` and `*_latest.pth`).
- Validation runs every 5 epochs; early stopping monitors combined SSIM + HU‑MAD.
- Mixed precision and gradient accumulation are enabled by default.

### 5. Inference & Evaluation

```bash
# Synthesize CECT from NCCT (A→B)
python test_infer.py --config configs/bihugan.yaml --direction A2B

# Synthesize NCCT from CECT (B→A)
python test_infer.py --config configs/bihugan.yaml --direction B2A

# Do not export DICOM (only compute metrics)
python test_infer.py --config configs/bihugan.yaml --direction A2B --no_export
```

**Output**:
- Console prints masked SSIM, PSNR, NMAE, and HU‑MAD.
- DICOM series are exported to `output_root/` with preserved spatial metadata and a “(BiHU‑GAN)” tag.

---

## 📊 Evaluation Metrics

The code computes radiotherapy‑relevant metrics directly on **HU‑space**:

| Metric | Description |
|--------|-------------|
| **SSIM** | Structural similarity (masked to body region) |
| **PSNR** | Peak signal‑to‑noise ratio (dB) |
| **NMAE** | Normalized mean absolute error (range `[-1,1]`) |
| **HU‑MAD** | Mean absolute deviation of Hounsfield Units (global / soft‑tissue / bone) |

For dosimetric evaluation (gamma pass rate, DVH differences) we used an external TPS (Varian Eclipse) – those scripts are not included but can be implemented using the exported DICOM series.

---

## 🧪 Core Innovations (Code Highlights)

### HU Consistency Loss

```python
# bihugan/losses/hu_loss.py
class HUConsistencyLoss(nn.Module):
    def forward(self, x_norm_ref, x_norm_pred, mask_body):
        hu_ref = inv_normalize_hu(x_norm_ref)
        hu_pred = inv_normalize_hu(x_norm_pred)
        # tissue‑weighted mask: soft tissue (1.0) vs bone (1.5)
        loss = weighted_mae(hu_pred, hu_ref, mask_body)
        return loss
```

### Gradient Consistency Loss

```python
# bihugan/losses/gradients.py
def gradient_consistency_loss(x_fake, x_real, grad_op):
    grad_fake = grad_op(x_fake)   # Sobel magnitude
    grad_real = grad_op(x_real)
    return F.l1_loss(grad_fake, grad_real)
```

### CBAM Integration

```python
# bihugan/models/generator.py
self.cbam1 = CBAM(base)   # after first conv
self.cbam2 = CBAM(base*2) # after first downsampling
...
```

---

## 📝 Important Notes for Users

- **Data privacy**: The repository does **not** include any patient data or institution‑specific preprocessing pipelines. You must adapt the DICOM path reading logic to your local storage structure.
- **Rigid registration**: The current dataset loader assumes that paired NCCT and CECT are already roughly aligned (same‑session acquisition). For unpaired or misaligned data, you need to add a registration step before feeding into the dataloader.
- **External resolution**: The model was trained on 1024×1024. If your data is 512×512, set `upsample_to_model: true` in the config – the loader will bilinearly upsample during training and inference.
- **MC Dropout**: Set `mc_dropout_p_train > 0` to enable Monte Carlo Dropout during training; set `mc_dropout_p_infer > 0` to sample multiple forward passes for uncertainty maps (not yet implemented in the test script – you can easily extend it).

---

## 📄 Citation & Contact

If you find this code inspiring for your research or wish to apply it in your work, we would greatly appreciate it if you could **cite our paper** once it is published (citation details will be provided upon acceptance).  

If you would like to learn more about our ongoing projects or discuss potential collaboration, please feel free to contact us at **18144845204@163.com**. Thank you very much for your support!

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE` for details.

