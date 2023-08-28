# Bayesian Flow Networks

A PyTorch implementation of [Bayesian Flow Networks](https://arxiv.org/abs/2308.07037)

Currently use of a non causal version of LLAMA2. Currently training TinyStories Models matching https://github.com/karpathy/llama2.c/tree/master


I am going to be using this repository to explore training dynamics of this new class of models. I will maintain a minimal implimentation in the Minimal.ipynd for a simple BFN implimentation. Everything else is an early work in progress.

![Paper Figure - BFN](https://github.com/Algomancer/Bayesian-Flow-Networks/blob/main/bfn.jpeg)

## Features

![Correctness](https://github.com/Algomancer/Bayesian-Flow-Networks/blob/main/correctness.png)

- [x] Discrete model with continuous-time loss, training and sampling (completed)
- [x] SOTA performance on XOR dataset
- [x] Tiny Stories 15m LLAMA2 Initial Code
- [ ] Tiny Stories weights (training)
- [ ] Wiki Text8 Dataset
- [ ] Bayesian Flow GPT-2 Scale
- [ ] Fancy Visuals
