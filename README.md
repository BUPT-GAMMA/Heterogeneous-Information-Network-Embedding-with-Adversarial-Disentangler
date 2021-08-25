# HEAD

Source code for TKDE 2021 paper ["**Heterogeneous Information Network Embedding with Adversarial Disentangler**"](https://ieeexplore.ieee.org/document/9483653)



## Environment Settings

* python == 3.7.11
* torch == 1.8.0



## Parameter Settings

Please refer to the **yaml file** of the corresponding dataset.

- model
  - vae: module architecture and training settings (e.g., learning rate) of the meta-path disentangler
  - D_mp: module architecture and training settings (e.g., learning rate) of the meta-path discriminator
  - D: module architecture and training settings (e.g., learning rate) of the semantic discriminator

- trainer
  - lambda (loss weight settings)
    - reconstruct: loss weight for reconstructing input node embedding
    - kl: loss weight for Kullback-Leibler Divergence in the meta-path disentangler
    - adv_mp_clf: loss weight for adversarial classification of meta-path discriminator
    - gp: loss weight for grad penalty
    - d_adv: loss weight for real/fake classifier
    - d_clf: loss weight for semantic classifier



## Files in the folder

~~~~
HEAD/
├── code/
│   ├── train_ACM.py: training the HEAD model on ACM
│   ├── train_Aminer.py: training the HEAD model on Aminer
│   ├── train_DBLP.py: training the HEAD model on DBLP
│   ├── train_Yelp.py: training the HEAD model on Yelp
│   ├── config/
│   │		├── ACM.yaml: parameter settings for ACM
│   │		├── Aminer.yaml: parameter settings for Aminer
│   │		├── DBLP.yaml: parameter settings for DBLP
│   │		└── Yelp.yaml: parameter settings for Yelp
│   ├── evaluate/
│   │		├── ACM_evaluate.py
│   │		├── Aminer_evaluate.py
│   │		├── DBLP_evaluate.py
│   │		└── Yelp_evaluate.py
│   ├── src/
│   │		├── bi_model.py: implementation of two meta-paths
│   │		├── tri_model.py: implementation of three meta-paths
│   │		├── data.py
│   │		└── tri_model.py
├── datasets/
└── README.md
~~~~



## Basic Usage

~~~
python train_DBLP.py ./config/DBLP.yaml
~~~



## Hyper-parameter Tuning

The architectures of three main modules make a great difference. Besides, there are three key hyper-parameters: *lr*, *kl* and *gp*.





# Reference

```
@article{wang2021heterogeneous,
  title={Heterogeneous Information Network Embedding with Adversarial Disentangler},
  author={Wang, Ruijia and Shi, Chuan and Zhao, Tianyu and Wang, Xiao and Ye, Yanfang Fanny},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```
