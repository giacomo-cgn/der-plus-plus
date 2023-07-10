# DER++ implementation
Implementation and test of DER++ Continual Learning baseline from the article _Dark Experience for General Continual Learning: a Strong, Simple Baseline_, _Buzzega et al._ (https://arxiv.org/pdf/2004.07211.pdf).

The method was implemented mainly using [Avalanche](https://github.com/ContinualAI/avalanche) and Pytorch libraries.
Tests and comparison of the model were done mainly on Split-Cifar100 dataset, which was not explored in the original paper

## Project structure
- `der.py` contains the implementation of DER/DER++ as an Avalanche plugin + a custom reservoir buffer.
- `preliminary_analysis_SCIFAR10.ipynb` contains a comparison on Split-CIFAR10 of DER++, Experience Replay and Naive (standard finetuning), trying to reproduce the orignal paper.
- `grid_search_der.ipynb` contains the grid search code and analysis over DER++ hyperparameters on Split-CIFAR100.
- `grid_search_replay_naive.ipynb` contains the grid search code over Experience Replay and Naive hyperparameters on Split-CIFAR100.
- `final_retrain` contains the retrain on the entire training stream of Split-CIFAR100 for DER++, Replay and Naive. Contains also comparisons between the 3 methods final results.
- `results` folder containing the results of the various model runs.
