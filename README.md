# Spatial Distillation based Distribution Alignment (SDDA) for Cross-Headset EEG Classification
This repository contains the original Python code for our paper [Spatial Distillation based Distribution Alignment (SDDA) for Cross-Headset EEG Classification](https://arxiv.org/pdf/2503.05349).

![SDDA_approach](https://github.com/Dingkun0817/SDDA/blob/main/SDDA_approach.jpg)

# Overview
A non-invasive brain-computer interface (BCI) enables direct interaction between the user and external devices, typically via electroencephalogram (EEG) signals. However, decoding EEG signals across different headsets remains a significant challenge due to differences in the number and locations of the electrodes. 

To address this challenge, we propose a spatial distillation based distribution alignment (SDDA) approach for heterogeneous cross-headset transfer in non-invasive BCIs. SDDA uses first spatial distillation to make use of the full set of electrodes, and then input/feature/output space distribution alignments to cope with the significant differences between the source and target domains. 

To our knowledge, this is the first work to use knowledge distillation in cross-headset transfers. Extensive experiments on six EEG datasets from two BCI paradigms demonstrated that SDDA achieved superior performance in both offline unsupervised domain adaptation and online supervised domain adaptation scenarios, consistently outperforming 10 classical and state-of-the-art transfer learning algorithms.

# Contributions
- We propose spatial distillation (SD) for heterogeneous transfer learning among different EEG headsets, leveraging knowledge from EEG signals with more channels to improve those with fewer channels. This approach effectively addresses the challenge of limited spatial information utilization inherent in fewer-channel headsets.
- We introduce a distribution alignment (DA) strategy that aligns the source and target domains comprehensively in multiple stages of the model, i.e., input/feature/output spaces. Unlike previous approaches that rely on single-stage alignment, the proposed DA more effectively bridges the domain gaps, ensuring robust transfer.
- Extensive experiments on multiple EEG datasets, covering both motor imagery (MI) and P300 paradigms, validated the superior performance of SDDA, which consistently outperformed state-of-the-art homogeneous transfer learning approaches in both offline and online calibration scenarios.

