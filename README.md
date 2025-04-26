# Spatial Distillation based Distribution Alignment (SDDA) for Cross-Headset EEG Classification
### ✨ This repository contains the original Python (PyTorch) code for our paper [Spatial Distillation based Distribution Alignment (SDDA) for Cross-Headset EEG Classification](https://arxiv.org/pdf/2503.05349).


# Overview
A non-invasive brain-computer interface (BCI) enables direct interaction between the user and external devices, typically via electroencephalogram (EEG) signals. However, decoding EEG signals across different headsets remains a significant challenge due to differences in the number and locations of the electrodes. 

To address this challenge, we propose a spatial distillation based distribution alignment (SDDA) approach for heterogeneous cross-headset transfer in non-invasive BCIs. SDDA uses first spatial distillation to make use of the full set of electrodes, and then input/feature/output space distribution alignments to cope with the significant differences between the source and target domains. 

To our knowledge, this is the first work to use knowledge distillation in cross-headset transfers. Extensive experiments on six EEG datasets from two BCI paradigms demonstrated that SDDA achieved superior performance in both offline unsupervised domain adaptation and online supervised domain adaptation scenarios, consistently outperforming 10 classical and state-of-the-art transfer learning algorithms.

# Contributions
- We propose spatial distillation (SD) for heterogeneous transfer learning among different EEG headsets, leveraging knowledge from EEG signals with more channels to improve those with fewer channels. This approach effectively addresses the challenge of limited spatial information utilization inherent in fewer-channel headsets.
- We introduce a distribution alignment (DA) strategy that aligns the source and target domains comprehensively in multiple stages of the model, i.e., input/feature/output spaces. Unlike previous approaches that rely on single-stage alignment, the proposed DA more effectively bridges the domain gaps, ensuring robust transfer.
- Extensive experiments on multiple EEG datasets, covering both motor imagery (MI) and P300 paradigms, validated the superior performance of SDDA, which consistently outperformed state-of-the-art homogeneous transfer learning approaches in both offline and online calibration scenarios.

# Proposed SDDA Framework
![SDDA_approach](https://github.com/Dingkun0817/SDDA/blob/main/SDDA_approach.jpg)

## Datasets 
   To download the BNCI public datasets, please follow the link below to access the preprocessed datasets.

   [Download BNCI Public Datasets](http://www.bnci-horizon-2020.eu/database/data-sets)

   For more detailed information about the datasets (e.g., experimental paradigms, subjects, sessions), please visit:
   
   [More details of the BNCI Datasets](https://neurotechx.github.io/moabb/datasets.html)

## Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

This code has been tested with `Python 3.10` and `3.11`.

`pip install -r requirements.txt`

## Baselines

In this paper, I have implemented some deep transfer methods as follows:
* [DAN](https://proceedings.mlr.press/v37/long15): Learning Transferable Features with Deep Adaptation Networks (ICML2015)
* [DANN](http://www.jmlr.org/papers/v17/15-239.html): Domain-Adversarial Training of Neural Networks (JMLR2016)
* [CDAN](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html): Conditional Adversarial Domain Adaptation (NIPS2018)
* [JAN](http://proceedings.mlr.press/v70/long17a.html): Deep Transfer Learning with Joint Adaptation Networks (PMLR2017)
* [MDD](https://proceedings.mlr.press/v97/zhang19i.html?ref=https://codemonkey): Bridging Theory and Algorithm for Domain Adaptation (PMLR2019)
* [MCC](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_28): Minimum Class Confusion for Versatile Domain Adaptation (ECCV2020)
* [SHOT](https://ieeexplore.ieee.org/abstract/document/9512429/): Source Data-Absent Unsupervised Domain Adaptation Through Hypothesis Transfer and Labeling Transfer (PAMI2021)
* [ISFDA](https://dl.acm.org/doi/abs/10.1145/3474085.3475487): Imbalanced Source-free Domain Adaptation (ACM MM2021)

If you wanna know the implement of the baselines, go [here](https://github.com/sylyoung/DeepTransferEEG)

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{liu2025spatial,
  title={Spatial distillation based distribution alignment (SDDA) for cross-headset EEG classification},
  author={Liu, Dingkun and Li, Siyang and Wang, Ziwei and Li, Wei and Wu, Dongrui},
  journal={arXiv preprint arXiv:2503.05349},
  year={2025}
}
```

## Contact
For any questions or collaborations, please feel free to reach out via d202481536@hust.edu.cn or open an issue in this repository.
