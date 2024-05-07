# Lighter Model for a Deeper Wallet: Quantizing and Pruning STAN Credit Card Fraud Detection Model

A Financial Fraud Detection Framework Optimized for Edge Devices

Columbia University COMS 6998: Applied ML on the Cloud Spring 2024

Authors: Samuel Braun, Steven Chase, Aditya Kankariya

Based on the paper and code by Dawei Cheng, Sheng Xiang, Chencheng Shang,
Yiyi Zhang, Fangzhou Yang, Liqing Zhang
- `STAN`: Spatio-temporal attention-based neural network for credit card fraud detection, in AAAI2020
- [GitHub](https://github.com/AI4Risk/antifraud)

## Usage

To use our implementations of `STAN` run
```
python main.py --mode <mode>
```

Modes that can be called with our model:
- `train`: Trains a model using the data in the data folder and the hyperparameters in the config file. Saves the model state dictionary to the path found in the config file
- `test`: Loads a pre-trained model from the path in the config file and evaluates its performace. Prints the performance metrics
- `prune`: Performs pruning on a pre-trained model loaded from the path in the config file. Will save the model to the file path in the config filie with "_pruned" added to the file name. The number of pruning iterations can be set in the config file.
- `quantize`: Will load a pre-trained model from the model path in the config file and will perform quantization. Will save the model to the file path in the config filie with "_quantized" added to the file name.

Configuration file can be found in `config/stan_cfg.yaml`

### Data Description

There are three datasets, YelpChi, Amazon and S-FFSD, utilized for model experiments in this repository.

YelpChi and Amazon datasets are from [CARE-GNN](https://dl.acm.org/doi/abs/10.1145/3340531.3411903), whose original source data can be found in [this repository](https://github.com/YingtongDou/CARE-GNN/tree/master/data).

S-FFSD is a simulated & small version of finacial fraud semi-supervised dataset. Description of S-FFSD are listed as follows:
|Name|Type|Range|Note|
|--|--|--|--|
|Time|np.int32|from $\mathbf{0}$ to $\mathbf{N}$|$\mathbf{N}$ denotes the number of trasactions.  |
|Source|string|from $\mathbf{S_0}$ to $\mathbf{S}_{ns}$|$ns$ denotes the number of transaction senders.|
|Target|string|from $\mathbf{T_0}$  to $\mathbf{T}_{nt}$ | $nt$ denotes the number of transaction reveicers.|
|Amount|np.float32|from **0.00** to **np.inf**|The amount of each transaction. |
|Location|string|from $\mathbf{L_0}$  to $\mathbf{L}_{nl}$ |$nl$ denotes the number of transacation locations.|
|Type|string|from $\mathbf{TP_0}$ to $\mathbf{TP}_{np}$|$np$ denotes the number of different transaction types. |
|Labels|np.int32|from **0** to **2**|**2** denotes **unlabeled**||
## Repo Structure
The repository is organized as follows:
- `models/`: the pre-trained models for each method. The readers could either train the models by themselves or directly use our pre-trained models;
- `data/`: dataset files;
- `config/`: configuration files for model;
- `feature_engineering/`: data processing;
- `methods/`: implementations of models;
- `main.py`: main driver of model;
- `requirements.txt`: package dependencies;

## Sources:

### Contributors :
<a href="https://github.com/AI4Risk/antifraud/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Risk/antifraud" />
</a>

### Citing

If you find *Antifraud* is useful for your research, please consider citing the following papers:

    @inproceedings{Xiang2023SemiSupervisedCC,
        title={Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation},
        author={Sheng Xiang and Mingzhi Zhu and Dawei Cheng and Enxia Li and Ruihui Zhao and Yi Ouyang and Ling Chen and Yefeng Zheng},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2023}
    }
    @article{cheng2020graph,
        title={Graph Neural Network for Fraud Detection via Spatial-temporal Attention},
        author={Cheng, Dawei and Wang, Xiaoyang and Zhang, Ying and Zhang, Liqing},
        journal={IEEE Transactions on Knowledge and Data Engineering},
        year={2020},
        publisher={IEEE}
    }
    @inproceedings{cheng2020spatio,
        title={Spatio-temporal attention-based neural network for credit card fraud detection},
        author={Cheng, Dawei and Xiang, Sheng and Shang, Chencheng and Zhang, Yiyi and Yang, Fangzhou and Zhang, Liqing},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={34},
        number={01},
        pages={362--369},
        year={2020}
    }
    @inproceedings{fu2016credit,
        title={Credit card fraud detection using convolutional neural networks},
        author={Fu, Kang and Cheng, Dawei and Tu, Yi and Zhang, Liqing},
        booktitle={International Conference on Neural Information Processing},
        pages={483--490},
        year={2016},
        organization={Springer}
    }
