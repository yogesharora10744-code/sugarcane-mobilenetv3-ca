# MobileNetV3Large with Coordinate Attention for Sugarcane Leaf Disease Classification

Official code for the paper:

> **MobileNetV3Large with Coordinate Attention for Accurate and Efficient Multi-Class Sugarcane Leaf Disease Classification**
> Priya Bhardwaj, Shekhar, Pritibha Sukhroop, Yogesh Kumar
> *Submitted to Pattern Analysis and Applications (Springer)*

The notebook reproduces every experiment in the paper. It covers a leakage-aware data pipeline that removes exact duplicates before splitting, eleven classifiers trained under one protocol, an attention ablation with exact McNemar tests, statistical and calibration analysis, and qualitative lesion segmentation.

## Dataset

- **Source:** [Sugarcane Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/akilesh253/sugarcane-plant-diseases-dataset)
- **Classes (6):** Bacterial Blight, Healthy, Mosaic, Red Rot, Rust, Yellow
- **Size:** 19,926 images, reduced to 19,086 after removing 840 exact (SHA-256) duplicates
- **Split:** stratified 70 / 15 / 15 (13,360 / 2,863 / 2,863 images), random seed 42

The dataset is not redistributed in this repository. The notebook downloads it with the Kaggle API.

## How to run

1. Open `sugarcane_leaf_disease_classification.ipynb` in [Google Colab](https://colab.research.google.com/) and select a GPU runtime (**Runtime → Change runtime type → GPU**).
2. Create a Kaggle API token (**Kaggle → Settings → API → Create New Token**) and upload `kaggle.json` when Section 1 asks for it.
3. Run all cells **in order** in one session. Each model section (9–17) reuses the split and the `tf.data` pipeline built in Sections 3–5.

The original experiments ran on an NVIDIA A100 GPU with TensorFlow 2.20.0. Outputs from that run are saved in the notebook.

## Notebook structure

| Section | Content |
|---|---|
| 1–2 | Dataset download and inspection |
| 3 | Image integrity screening and SHA-256 duplicate removal |
| 4 | Label encoding and stratified 70/15/15 split |
| 5 | `tf.data` pipeline and online augmentation |
| 6 | Custom CNN baseline |
| 7 | EfficientNetB0 (frozen) |
| 8 | Statistical and calibration comparison: custom CNN vs EfficientNetB0 |
| 9–16 | ResNet50, DenseNet121, MobileNetV3Large, EfficientNetV2B0, ResNet50-CBAM, DenseNet121-CBAM, MobileNetV3Large-ECA, ConvNeXtTiny |
| 17 | **Proposed MobileNetV3Large + Coordinate Attention** |
| 18 | Qualitative leaf extraction and lesion segmentation (visualisation only) |
| 19 | Attention ablation study with exact McNemar tests |
| 20 | Additional analysis: McNemar test of MobileNetV3Large-CA vs ResNet50, and ResNet50 probabilistic metrics |

## Main results (test set, n = 2,863)

| Model | Params (M) | Accuracy (%) | Macro F1 (%) | MCC |
|---|---:|---:|---:|---:|
| ResNet50 | 24.12 | 92.94 | 92.22 | 0.9149 |
| ResNet50-CBAM | 24.65 | 92.60 | 91.84 | 0.9104 |
| **MobileNetV3Large-CA (proposed)** | **3.34** | **91.79** | **90.89** | **0.9009** |
| MobileNetV3Large-ECA | 3.25 | 90.57 | 89.60 | 0.8859 |
| MobileNetV3Large | 3.25 | 90.50 | 89.44 | 0.8853 |
| ConvNeXtTiny | 28.02 | 90.19 | 89.05 | 0.8815 |
| DenseNet121-CBAM | 7.44 | 89.21 | 88.07 | 0.8696 |
| DenseNet121 | 7.31 | 89.10 | 87.95 | 0.8684 |
| EfficientNetV2B0 | 6.25 | 88.72 | 87.42 | 0.8642 |
| EfficientNetB0 (frozen) | 4.38 | 87.46 | 86.07 | 0.8490 |
| Custom CNN | 0.42 | 84.60 | 82.68 | 0.8149 |

**Attention ablation:** Coordinate Attention was the only module that significantly improved its backbone. It added 1.29 percentage points of accuracy to MobileNetV3Large (exact McNemar p = 0.003) for a 2.72 % parameter overhead. CBAM on ResNet50 (p = 0.353), CBAM on DenseNet121 (p = 0.872) and ECA on MobileNetV3Large (p = 0.930) showed no significant effect.

## Requirements

Google Colab provides everything except `kaggle`, which the notebook installs itself. For a local run:

```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the paper (citation details will be updated after publication):

```bibtex
@article{bhardwaj2026sugarcane,
  title   = {MobileNetV3Large with Coordinate Attention for Accurate and Efficient Multi-Class Sugarcane Leaf Disease Classification},
  author  = {Bhardwaj, Priya and Shekhar and Sukhroop, Pritibha and Kumar, Yogesh},
  journal = {Pattern Analysis and Applications},
  year    = {2026},
  note    = {Submitted}
}
```

## Contact

Yogesh Kumar (corresponding author), Department of Computer Science & Engineering, School of Technology, Pandit Deendayal Energy University, Gandhinagar, India. Email: yogesh.arora10744@gmail.com

## License

The code is released under the MIT License (see `LICENSE`). The dataset remains under its own Kaggle licence (CDLA-Sharing-1.0).
