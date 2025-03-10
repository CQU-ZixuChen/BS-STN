## **Description**

This repository includes the source codes of BS-STN and the sliding bearing dataset collected by Chongqing University.
The training codes and demos of BS-STN will be released soon...

## **Self-collected sliding bearing dataset**

$\bullet$ [Sliding bearing dataset](https://drive.google.com/drive/folders/19l1M2ZZWQtwTa73PR6RlN_sVvHrcLwi_?usp=drive_link)

![image](https://github.com/CQU-ZixuChen/BS-STN/blob/main/SlidingBearingTestBench.png)


## **Related works**

$\bullet$ [Bi-structural spatial–temporal network for few-shot fault diagnosis of rotating machinery](https://www.sciencedirect.com/science/article/pii/S0888327025000792)

$\bullet$ [Dynamic characteristics of tilting-pad coupled-bearing-rotor systems considering mixed lubrication and pad wear](https://www.sciencedirect.com/science/article/abs/pii/S0301679X24000380)

## **Citation**

@article{CHEN2025112378,
title = {Bi-structural spatial–temporal network for few-shot fault diagnosis of rotating machinery},
journal = {Mechanical Systems and Signal Processing},
volume = {227},
pages = {112378},
year = {2025},
issn = {0888-3270},
doi = {https://doi.org/10.1016/j.ymssp.2025.112378},
url = {https://www.sciencedirect.com/science/article/pii/S0888327025000792},
author = {Zixu Chen and J.C. Ji and Qing Ni and Benyuan Ye and Xiaoxi Ding and Wennian Yu},
keywords = {Bi-structural, Spatial–temporal information, Adaptive inference fusion, Fault diagnosis, Sliding bearing dataset, Rotating machinery},
abstract = {Recently, many spatial–temporal diagnostic frameworks have been proposed to improve the few-shot diagnostic performance of rotating machinery through extracting inherent and generalized features. But most methods extract spatial and temporal features separately by stacking 1D convolution and temporal modules, which could decrease the spatial–temporal correlation and result in the loss of information. The emerging graph convolutional network (GCN) is adept at handling multi-channel data, providing a more powerful spatial feature extraction method. Therefore, it is worth extending spatial–temporal frameworks by incorporating GCN. However, the graphs in most GCN-based methods are constructed through single modeling strategy such as KNN-Graph or Radius-Graph, which may not exhibit high generalization in dealing with different datasets. To address these limitations and improve the diagnostic performance of rotating machinery in few-shot scenarios, a Bi-Structural Spatial-Temporal Network (BS-STN) is proposed in this paper. Physic and function informed graphs with bi-structural inference paths, are introduced to obtain richer information through feature fusion. An adaptive inference fusion module is designed to dynamically adjust the contribution of fusion-view and single-view features to gradient descent, thereby avoiding potential overfitting caused by feature fusion. Multiple time-step graphs are connected by the same nodes to construct a temporal graph, which in combination with broadcast-based temporal and spatial embedding enable graph convolution to propagate temporal and spatial information synchronously, thus avoiding information loss and extracting inherent features. A thrust sliding bearing dataset is open-sourced, comprising multi-channel signals of normal, thrust pad wear and lubricating oil contamination. Experiments are conducted on two publicly available datasets and the self-collected sliding bearing dataset. The comparative analyses in different few-shot scenarios demonstrate the effectiveness and superiority of BS-STN. The source codes of the proposed method and the self-collected sliding bearing data are available at: https://github.com/CQU-ZixuChen/BS-STN.}
}

@article{ZHANG2024109287,
title = {Dynamic characteristics of tilting-pad coupled-bearing-rotor systems considering mixed lubrication and pad wear},
journal = {Tribology International},
volume = {192},
pages = {109287},
year = {2024},
issn = {0301-679X},
doi = {https://doi.org/10.1016/j.triboint.2024.109287},
url = {https://www.sciencedirect.com/science/article/pii/S0301679X24000380},
author = {Chaodong Zhang and Wennian Yu and Lu Zhang},
keywords = {Tilting-pad coupled bearing-rotor system, Mixed-lubrication dynamic model, Pad wear, Dynamic characteristics},
abstract = {The coupled bearing (including a tilting-pad journal and a thrust bearing) is mainly used in a nuclear power circulating pump to simultaneously support the high axial loads and radial loads of a vertical rotary system with minimum power loss, low vibration, and high load capacity. Its lubrication and vibration characteristics have significant effects on the operation reliability of the nuclear power circulating pump. In this paper, an original mixed-lubrication dynamic model for the coupled bearing-rotor system is proposed to study the dynamic characteristics of the system considering the effects of the pivot clearance, asperity contact, and elastic deformation. The innovation of the proposed model is that it integrates the horizontal-rocking vibrations of the rotor and pads with the mixed lubrication of the coupled bearing and the wear of journal pads and thrust pads. The effects of different external loads and pad wear levels on the dynamic characteristics of the coupled bearing-rotor system are revealed. Additionally, a series of experiments are performed to measure the vibration responses of the system, which are directly compared to the simulated responses of the proposed model for validation purposes. It is expected that the proposed mixed-lubrication dynamic model can provide practical engineering guidance for the condition monitoring and fault identification of coupled bearing-rotor systems.}
}
