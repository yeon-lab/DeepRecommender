# Deep Learning based Recommender System using Cross Convolutional Filters

## Introduction
This repository contains source code for paper "[Deep learning based recommender system using cross convolutional filters](https://www.sciencedirect.com/science/article/pii/S0020025522000561)", *Information Sciences*.

With the recent development of online transactions, recommender systems have increasingly attracted attention in various domains. The recommender system supports the users' decision-making by recommending items that are more likely to be preferred. Many studies in the field of deep learning-based recommender systems have attempted to capture the complex interactions between users' and items' features for accurate recommendation.

In this paper, we propose a recommender system based on the convolutional neural network using the outer product matrix of features and cross-convolutional filters. The proposed method can deal with the various types of features and capture the meaningful higher-order interactions between users and items, giving greater weight to important features. Moreover, it can alleviate the overfitting problems since the proposed method includes the global average or max-pooling instead of the fully connected layers in the structure. Experiments showed that the proposed method performs better than the existing methods by capturing important interactions and alleviating the overfitting issue.



## Model
![Figure2_org](https://github.com/yeon-lab/DeepRecommender/assets/39074545/6e836393-5c6f-4d47-8be7-90ebd8f73553)



## Installation

Numpy, scikit-learn, and PyTorch (CUDA toolkit if use GPU)



## Usage

### Training and test
```python 
```

### Hyper-parameters
Hyper-parameters are set in config.json
>
* `nhid`: a list of the number of hidden channels in convolutional layers. E.g., [32, 64, 128]
* `pool`: a type of global fooling, {'avg','max'}.
* `emb`: embedding dimension.
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterian for early stopping. The first word is 'min' or 'max', the second one is metric.


