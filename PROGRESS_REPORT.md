# ROB 472 Winter 2026 — Underwater Danger Map: Progress Report

**Brandon McDonald, Caitlin Roberts, Sydney Ragla**
University of Michigan · ROB 472 Winter 2026

---

## Introduction

Marine robots operating underwater face more perception challenges compared to terrestrial systems. Light scattering, color attenuation, turbidity, and low visibility make it difficult for autonomous underwater vehicles (AUVs) to reliably interpret and detect objects in their surroundings [1]. Yet accurate perception is essential for tasks such as shipwreck inspection, environmental monitoring, coral reef surveys, and infrastructure inspection. Effective autonomy in underwater environments requires robots to both interpret the semantic content of their environment and estimate spatial depth of surrounding objects [2]. For example, an AUV performing close-range imaging must avoid collisions with the seafloor, reefs, debris, or other obstacles while navigating in cluttered conditions. This requires real-time segmentation and distance estimation, enabling safer autonomy.

Models for underwater semantic segmentation and monocular depth estimation have been recently developed to address these challenges. The Semantic Segmentation of Underwater Imagery (SUIM) dataset is the first large-scale dataset for underwater image segmentation [3]. The SUIM dataset, along with the SUIM-Net segmentation model, provide eight classification categories that improve understanding of underwater scenes across a range of environments. The SUIM dataset includes object classes for fish, reefs, plants, and wrecks or ruins, all of which are highly relevant to underwater exploration and surveying [3].

At the same time, the SPADE algorithm provides monocular depth estimation for real-time underwater applications [4]. The SPADE algorithm improves on Ebner's previous work in monocular depth estimation as it achieves competitive performance at significantly faster runtimes [4][5]. The SPADE monocular depth estimation pipeline demonstrates strong robustness to different levels of depth point sparsity and exhibits great generalization on underwater data [4].

The SUIM and SPADE algorithms, when used independently, are both well-suited for underwater vehicles. Our major project goal is to explore how combining underwater semantic segmentation and monocular depth estimation may improve environment understanding for underwater vehicles. Using these two models together, we will construct a depth-aware "danger map" that highlights nearby obstacles based on both type and estimated proximity to the robot.

---

## Technical Approach

Our technical approach consists of three main parts: underwater semantic segmentation, monocular depth estimation, and combining these results into a danger map that reflects collision risk.

### Benchmark 1: SUIM-Net Semantic Segmentation

**What we did.** We evaluated the SUIM-Net model (Islam et al., 2020) [3], a fully-convolutional encoder-decoder network trained on the SUIM dataset. The model classifies each pixel into five obstacle-relevant categories: Fish/Vertebrates (FV), Human Divers (HD), Reef/Invertebrates (RI), Robot/Instruments (RO), and Wrecks/Ruins (WR). We ran inference on the SUIM test split to establish an in-domain baseline, then evaluated cross-dataset generalization on DeepFish [6] and USIS10K [7]. Per-pixel metrics (IoU, Dice, Precision, Recall) were computed per class for every image.

**Datasets tested.**

| Dataset | Images | Classes evaluated | Type |
|---------|--------|-------------------|------|
| SUIM TEST | 110 | FV, HD, RI, RO, WR | In-domain baseline |
| DeepFish | 310 | FV only | Cross-dataset generalization |
| USIS10K TEST | 1,596 | FV, HD, RI, RO, WR | Cross-dataset generalization |

**Results.**

*SUIM TEST (in-domain baseline):*

| Class | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| Fish/Vertebrate (FV) | 0.699 | 0.748 | 0.772 | 0.907 |
| Human Diver (HD) | 0.828 | 0.855 | 0.874 | 0.944 |
| Reef/Invertebrate (RI) | 0.602 | 0.632 | 0.646 | 0.948 |
| Robot/Instrument (RO) | 0.950 | 0.962 | 0.990 | 0.959 |
| Wreck/Ruin (WR) | 0.817 | 0.838 | 0.859 | 0.956 |
| **mIoU** | **0.779** | | | |

The model performs strongly on its training distribution. Robot/instrument detection is near-perfect (IoU = 0.950), and divers and wrecks are segmented reliably. Reef/invertebrate is the weakest class (IoU = 0.602), likely due to the high visual diversity of coral and invertebrate textures.

**Per-class metric breakdown (SUIM TEST):**

![SUIM TEST — Per-Class Mean Metrics](figures/charts/suim_class_means.png)

![SUIM TEST — IoU Distribution per Class](figures/charts/suim_iou_boxplots.png)

*DeepFish (cross-dataset, fish class only):*

| Class | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| Fish/Vertebrate (FV) | 0.061 | 0.084 | 0.631 | 0.063 |

Generalization to DeepFish drops sharply. The model rarely activates the fish class on DeepFish imagery despite high precision when it does (0.631), indicating it is highly conservative — it nearly never predicts fish (recall = 0.063). DeepFish images show fish against very different backgrounds (open water, aquaculture pens) compared to SUIM's reef-heavy scenes, which likely explains the failure.

*USIS10K TEST (cross-dataset):*

| Class | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| Fish/Vertebrate (FV) | 0.323 | 0.388 | 0.530 | 0.679 |
| Human Diver (HD) | 0.766 | 0.776 | 0.815 | 0.943 |
| Reef/Invertebrate (RI) | 0.184 | 0.213 | 0.240 | 0.829 |
| Robot/Instrument (RO) | 0.949 | 0.949 | 0.976 | 0.973 |
| Wreck/Ruin (WR) | 0.407 | 0.411 | 0.460 | 0.929 |
| **mIoU** | **0.526** | | | |

USIS10K shows partial generalization. Robot detection remains near-perfect (IoU = 0.949) and human divers transfer well (IoU = 0.766). Fish (0.323), Wreck (0.407), and particularly Reef (0.184) degrade significantly, with low precision indicating high false-positive rates. High recall for these classes (0.68–0.83) suggests the model is detecting something in the right regions but with poor boundary accuracy.

**Sample segmentation outputs:**

The figures below show SUIM-Net predictions on sample images. Each pixel is color-coded by class: Red = Robot, Yellow = Fish, Blue = Diver, Magenta = Reef, Cyan = Wreck.

| | |
|:---:|:---:|
| ![Diver + Robot](figures/suimnet/d_r_47_sidebyside.png) | ![Wreck + Diver](figures/suimnet/w_r_147_sidebyside.png) |
| Diver holding a robot instrument — HD (blue) and RO (red) correctly segmented, reef (magenta) along the seafloor | Wreck scene — WR (cyan) dominates the propeller blades, HD (blue) detected in the background |
| ![Fish + Reef](figures/suimnet/f_r_329_sidebyside.png) | ![Fish + Reef](figures/suimnet/n_l_100_sidebyside.png) |
| Sea turtle with diver — FV (yellow) on the turtle, HD (blue) on the diver, RI (magenta) on the reef below | Angelfish near coral — FV (yellow) cleanly segments the fish, RI (magenta) on the reef |

**Cross-dataset metric comparison:**

![DeepFish — Per-Class Mean Metrics](figures/charts/deepfish_class_means.png)

![USIS10K — Per-Class Mean Metrics](figures/charts/usis10k_class_means.png)

**Summary.** SUIM-Net establishes a strong in-domain baseline (mIoU = 0.779) but generalizes unevenly. Structural classes like robots and divers transfer well across datasets, while biotic classes (fish, reef) are highly scene-dependent. This is an important limitation for the danger map, as fish and reef are among the most safety-relevant objects for AUV navigation.

---

### Benchmark 2: SPADE Monocular Depth Estimation

**What we did.** We benchmarked the SPADE model (Zhang et al., 2025) [4], a two-stage depth estimation pipeline that combines a Depth Anything V2 monocular backbone with a Deformable Attention Transformer (DAT) refinement head conditioned on sparse depth hint points. SPADE produces dense, metric-scale depth maps from a single RGB image plus a small set of sparse depth measurements (typically from a sonar or SLAM system).

The official pretrained SPADE checkpoint is hosted on a restricted Google Drive and is not publicly accessible. We therefore assembled a runnable checkpoint from the publicly available Depth Anything V2 ViT-S backbone [8], with the DAT refinement head randomly initialized. This configuration corresponds to the **DA V2 + GA** (Global Alignment) baseline row in the SPADE paper — the full SPADE refinement head is not active. Sparse depth hints were simulated from dense ground-truth depth using Shi-Tomasi corner detection, consistent with the SPADE evaluation protocol.

**Datasets tested.**

| Dataset | Images | GT depth source | Evaluation ranges |
|---------|--------|-----------------|-------------------|
| FLSea-VI validation | 4,483 | Depth image (metres) | ≤10 m, ≤5 m, ≤2 m |
| SeaThru | ~1,100 | SFM reconstruction (metres) | ≤10 m, ≤5 m, ≤2 m |

**Results.** Quantitative SPADE results are pending — evaluation jobs are currently running on the ARC Great Lakes GPU cluster. Metric charts (error bars and δ-accuracy by depth range) will be generated automatically once the metrics CSV is produced. Based on the DA V2 + GA configuration used, we expect performance in the range reported by the SPADE paper for that baseline (FLSea: MAE ≈ 0.277 m, RMSE ≈ 0.563 m, AbsRel ≈ 0.081 at ≤10 m) [4].

**Sample depth predictions (FLSea-VI):**

Each panel shows the RGB input with sparse depth hint points (left) and the predicted dense depth map (right). The colorbar indicates depth in metres.

| | |
|:---:|:---:|
| ![FLSea depth sample 1](figures/spade/flsea_001833.png) | ![FLSea depth sample 2](figures/spade/flsea_000647.png) |
| Seafloor with structured terrain — depth gradients captured from near (~3 m) to far (~12 m) | Sandy slope with fish — near-field seafloor and far-field water column distinguished |
| ![FLSea depth sample 3](figures/spade/flsea_003122.png) | |
| Rocky reef — smooth depth transition across foreground rocks to background water |

**Metrics reported.** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Absolute Relative Error (AbsRel), Scale-Invariant Log Error (SILog), and δ-accuracy thresholds (δ < 1.25, 1.25², 1.25³), all computed over valid pixels within each depth range.

---

## Remaining Milestones

| Sub-goal | Description | Target date |
|----------|-------------|-------------|
| SPADE benchmark results | Collect and analyze FLSea and SeaThru evaluation outputs once cluster jobs complete | Mar 2026 |
| Develop danger map algorithm | Combine SUIM-Net segmentation masks and SPADE depth maps into a per-pixel risk score. Each class is assigned a base hazard weight; risk is scaled by inverse depth so nearby objects dominate | Mar 30, 2026 |
| Test danger map | Run the fused danger map on held-out underwater sequences. Evaluate qualitatively (do high-risk regions correspond to nearby obstacles?) and quantitatively where ground truth is available | Apr 12, 2026 |
| Final report | Write final report summarizing all results | Apr 21, 2026 |

---

## References

[1] D. Q. Huy et al., "Object perception in underwater environments: a survey on sensors and sensing methodologies," *Ocean Engineering*, vol. 267, p. 113202, Jan. 2023. https://doi.org/10.1016/j.oceaneng.2022.113202

[2] B. Yu, J. Wu, and M. J. Islam, "UDepth: Fast Monocular Depth Estimation for Visually-guided Underwater Robots," arXiv:2209.12358, 2022. https://arxiv.org/abs/2209.12358

[3] M. J. Islam et al., "Semantic Segmentation of Underwater Imagery: Dataset and Benchmark," arXiv:2004.01241, 2020. https://arxiv.org/abs/2004.01241

[4] H. Zhang, G. Billings, and S. B. Williams, "SPADE: Sparsity Adaptive Depth Estimator for Zero-Shot, Real-Time, Monocular Depth Estimation in Underwater Environments," arXiv:2510.25463, 2025. https://arxiv.org/abs/2510.25463

[5] L. Ebner, G. Billings, and S. Williams, "Metrically Scaled Monocular Depth Estimation through Sparse Priors for Underwater Robots," arXiv:2310.16750, 2023. https://arxiv.org/abs/2310.16750

[6] S. Saleh et al., "DeepFish: A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis," *Scientific Reports*, 2020. https://alzayats.github.io/DeepFish/

[7] L. Zust et al., "LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark," ICCV 2023. https://lojzezust.github.io/lars-dataset/

[8] L. Yang et al., "Depth Anything V2," arXiv:2406.09414, 2024. https://arxiv.org/abs/2406.09414
