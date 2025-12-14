# Benchmarking Deep Anomaly Detection Models for Industrial Defect Detection

## Overview

This project presents a systematic benchmarking and comparison of deep-learning–based anomaly detection models, with a primary focus on GANomaly and PatchCore. Both models are evaluated under a unified experimental pipeline for unsupervised industrial defect detection.

The experiments are conducted using the MVTec Anomaly Detection dataset and are designed to be directly applicable to additive manufacturing inspection tasks such as surface defect detection, porosity identification, and process-induced anomalies.

---

## Models Compared

### GANomaly

GANomaly is a GAN-based anomaly detection approach that models normal data distributions via reconstruction in a latent space. Anomalies are identified using reconstruction and feature-level discrepancies.

Key characteristics:
- Generative, reconstruction-based method
- Sensitive to training stability and hyperparameters
- Limited localization performance for small defects

---

### PatchCore

PatchCore is a feature-embedding and nearest-neighbor–based anomaly detection method that builds a memory bank of normal patch-level features and detects anomalies via distance-based scoring.

Key characteristics:
- No adversarial or gradient-based training
- Strong performance on small and localized defects
- High reproducibility and industrial robustness

---

## Dataset

The experiments use the MVTec Anomaly Detection dataset, which is a standard benchmark for industrial anomaly detection. Each category contains only normal samples during training, while test sets include both normal and defective samples.

Example categories include bottle, cable, metal_nut, screw, zipper, and others.

---

## Experimental Setup

- Framework: anomalib (v2.x)
- Backbone network (PatchCore): ResNet-18
- Hardware: CUDA-enabled GPU
- Training paradigm: Unsupervised / one-class learning
- Evaluation metrics:
  - Image-level AUROC
  - Pixel-level AUROC (when applicable)
  - Inference stability and memory usage

All models are evaluated using the same preprocessing and evaluation pipeline to ensure a fair comparison.

---

## Key Findings

GANomaly exhibits moderate anomaly detection performance but suffers from training instability and weaker localization, particularly for subtle industrial defects.

PatchCore consistently achieves higher anomaly detection accuracy, better defect localization, and significantly improved robustness, making it more suitable for real-world industrial and additive manufacturing inspection scenarios.

---

## Reproducibility

The project emphasizes reproducibility through:
- deterministic inference pipelines
- consistent preprocessing across models
- memory-safe configurations for limited GPU environments
- explicit handling of dataset availability constraints

PatchCore experiments do not rely on gradient-based training, allowing fast and stable reruns.

---

## Project Structure

The repository is organized into logical components for data handling, model execution, and result analysis. It includes separate experiment folders for GANomaly and PatchCore, result directories for metrics and visualizations, and notebooks for benchmarking and analysis.

---

## Applicability to Additive Manufacturing

Although the benchmark uses MVTec AD, the conclusions are directly transferable to additive manufacturing defect detection tasks, including surface cracks, porosity, irregular melt patterns, and layer-wise inconsistencies.

PatchCore is particularly well-suited for such applications due to its patch-level modeling and strong localization capability.

---

## References

Akcay et al., GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training, ACCV 2018  
Roth et al., Towards Total Recall in Industrial Anomaly Detection, CVPR 2022  
Bergmann et al., The MVTec Anomaly Detection Dataset, CVPR 2019  

---

## License

This repository is intended for research and educational purposes. Dataset usage is subject to the original MVTec Anomaly Detection license.

---

## Acknowledgements

This work builds upon the anomalib framework and the broader research community focused on industrial anomaly detection.
