<h2 align="center">Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies</h2>

### 1. Overview
Code for the paper [Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies](https://openreview.net/forum?id=O453PHSthc) as presented in NeurIPS 2023. Gradient instability problems have been ever-present in reucrrent neural network training for decades. Solutions that tackle this problem often involve artificial elements such as gating mechanisms. As such, these solutions cannot be adopted in RNN models that seek to draw comparisons with the brain. This paper presents multiple approaches that involve specialized skip connections through time to support stable RNN training. All models are built on Tensorflow 2.10.

### 2. Methods
<p align="center">
  <img src="/figures/control.png" width="800">
</p>

<p align="center">
  <img src="/figures/cd.png" width="800">
</p>

<p align="center">
  <img src="/figures/sctt.png" width="800">
</p>

<p align="center">
  <img src="/figures/dasc.png" width="800">
</p>

### 3. Code


### 4. Citation

```
@inproceedings{soo2023temporalskip,
 author = {Soo, Wayne W.M. and Goudar, Vishwa and Wang, Xiao-Jing},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {TBD},
 pages = {TBD},
 publisher = {Curran Associates, Inc.},
 title = {Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies},
 url = {TBD},
 volume = {TBD},
 year = {2023}
}
```
