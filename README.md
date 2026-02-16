# M$^2$FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting

## Abstract
Forecasting time series with extreme events is difficult because such events are sparse, high-variance, and temporally irregular, causing large real-world errors for models that mainly capture regular patterns. We propose **MFMoE**, an extreme-adaptive forecasting model that learns regular and extreme dynamics via multi-resolution, multi-view frequency modeling. MFMoE uses a multi-view frequency mixture-of-experts over Fourier and wavelet spect`ra, with a shared band splitter to align partitions and encourage cross-view collaboration. A hierarchical adaptive fusion module aggregates coarse-to-fine frequency features, and a temporal gating module balances long-term trends with short-term frequency-aware signals. Experiments on hydrological datasets show MFMoE outperforms strong baselines without extreme-event labels.

## Acknowledgements
We thank the authors of the following repositories for their open-source code or datasets used in our experiments:

- MC-ANN: https://github.com/davidanastasiu/mcann  
- FreqMoE: https://github.com/sunbus100/FreqMoE-main
- Time-Series-Library: https://github.com/thuml/Time-Series-Library  
- MoLE: https://github.com/RogerNi/MoLE  

## Datasets
Datasets can be found in the MC-ANN repository:  
https://github.com/davidanastasiu/mcann

## Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{huang2026m2fmoe_aaai,
  title     = {M$^2$FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting},
  author    = {Yaohui Huang and Runmin Zou and Yun Wang and Laeeq Aslam and Ruipeng Dong},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence},
  year      = {2026}
}

@misc{huang2026m2fmoe_arxiv,
  title         = {M$^2$FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting},
  author        = {Yaohui Huang and Runmin Zou and Yun Wang and Laeeq Aslam and Ruipeng Dong},
  year          = {2026},
  eprint        = {2601.08631},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2601.08631}
}
