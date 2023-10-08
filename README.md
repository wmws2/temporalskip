<h2 align="center">Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies</h2>

### 1. Overview
Code for the paper [Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies](https://openreview.net/forum?id=O453PHSthc) as presented in NeurIPS 2023. Gradient instability problems have been ever-present in recurrent neural network training for decades. Solutions that tackle this problem often involve artificial elements such as gating mechanisms. As such, these solutions cannot be adopted in RNN models that seek to draw comparisons with the brain. This paper presents multiple approaches that involve specialized skip connections through time to support stable RNN training. All models are built with Tensorflow 2.10. In this work, we study continuous-time leaky RNNs described by:

```math
\mathbf{T}\frac{d\mathbf{r}}{dt} = -\mathbf{r} + f\left( \mathbf{W}_\text{rec}\mathbf{r} + \mathbf{b} + \mathbf{h}_\text{ext} + \boldsymbol{\eta} \right)
```
### 2. Methods

**2.1 Base model** 
<p align="center">
  <img src="/figures/control.png" width="600">
</p>
The most common way of training biologically plausible RNNs involves simulating its dynamics using Euler's method for the forward pass:

```math
\mathbf{r}_{t+\Delta t} \leftarrow \mathbf{r}_{t} + \frac{d\mathbf{r}_t}{dt} \Delta{t}
```
Consequently, iterating this over $\theta$ time steps results in:

```math
    \mathbf{r}^\text{base}_{t+\theta\Delta t} \leftarrow \mathbf{r}_{t} + \sum_{k=0}^{\theta-1}\frac{d\mathbf{r}_{t + k\Delta t}}{dt} \Delta{t}
```

**2.2 Coarsened discretization (CD)** 
<p align="center">
  <img src="/figures/cd.png" width="600">
</p>
CD begins by training the network with a large step size to support stable gradients while learning long-term dependencies:

```math
    \mathbf{r}^\text{CD}_{t+\theta\Delta t} \leftarrow \mathbf{r}_{t} + \frac{d\mathbf{r}_t}{dt} \Delta{t} \times \theta
```

for some discretization factor $\theta > 1$. Thereafter, $\theta$ is gradually reduced over the course of training until $\theta = 1$ at the end of training, resulting in a reversion to the base model. This coarsening process facilitates gradient stability during training by reducing the number of time steps needed to be backpropagated. 

**2.3 Skip connection through time (SCTT)** 
<p align="center">
  <img src="/figures/sctt.png" width="600">
</p>

SCTT similarly confines the well-established idea of skip connections through time to training only. For some mixing ratio $\beta$, a skip connection between time steps $t$ and $t+\theta$ alters the hidden state as:

```math
    \mathbf{r}^\text{SCTT}_{t+\theta\Delta t} \leftarrow (1-\beta) \, \mathbf{r}_{t} + \beta \, \mathbf{r}^\text{base}_{t+\theta\Delta t}
```

The skip length $\theta$ remains unchanged throughout training, while the mixing ratio $0 < \beta \leq 1$ starts at a small value. This provides shortcuts for gradient backpropagation, thus reducing the risk of gradient instability. Similar to CD, the mixing is gradually reduced (i.e. $\beta$ is gradually increased to 1) until we are left with a trained network without skip connections.

**2.4 Dynamics aligned skip connection (DASC)** 
<p align="center">
  <img src="/figures/dasc.png" width="600">
</p>

We propose an additional method, DASC, which combines the essence of CD with SCTT, in order to respect dynamical properties of the system. DASC achieves this by simulating large time-steps through its skip connections. The skip connections between $t$ and $t+\theta$ alters the hidden state as:

```math
    \mathbf{r}^\text{DASC}_{t+\theta\Delta t} \leftarrow (1-\beta) \, \mathbf{r}^\text{CD}_{t+\theta\Delta t} + \beta \, \mathbf{r}^\text{base}_{t+\theta\Delta t}
```

DASC aims to combine the benefits of CD and SCTT while limiting the risk of gradient instability. Unlike CD, DASC simultaneously trains the model with large and appropriately small time steps; unlike SCTT, it mitigates misalignment in the network's dynamics when it mixes hidden state estimates through the paths with and without skip connections. 

### 3. Code

**3.1 Function files** 

<code>func_init.py</code> <br>
* <code>initconstants</code> creates a dictionary of simulation constants <br>
* <code>compute_dueta</code> simulates the dynamics of the RNN by one time step <br>
* <code>compute_uall</code> simulates the dynamics of the RNN across a prefined number of time steps with options to include SCTT or DASC <br>

<code>func_standardtasks.py</code> contains a class which implements the main tasks of the paper (<code>X</code> = <code>A</code>, <code>B</code> or <code>C</code>)<br>
* <code>construct_ringinput</code> constructs the RNN input given the angle and modality of the stimulus <br>
* <code>construct_ringangle</code> computes the readout angle based on the RNN output <br>
* <code>construct_trialX</code> constructs the RNN inputs and targets for tasks with trial structure <code>X</code> <br>
* <code>trainX</code> trains the RNN to perform a task with trial structure <code>X</code> <br>
* <code>evalX</code> computes the true performance of the RNN on a task with trial structure <code>X</code> <br>
* <code>uallX</code> records the neural activity of the RNN for some specific task <br>

<code>func_rulereversaltask.py</code> contains a class which implements the rule reversal task in the later sections of the paper <br>
* <code>construct_trial</code> constructs the RNN inputs and targets of the task <br>
* <code>train</code> trains the RNN to perform the task <br>
* <code>eval</code> computes the true performance of the RNN on the task <br>

**3.1 Main files**  <br>
* <code>main_standardtasks.py</code> trains RNNs on the main tasks of the paper using the base model, SCTT and DASC  <br>
* <code>main_standardtasks_cd.py</code> trains RNNs on the main tasks of the paper using CD  <br>
* <code>main_rulereversaltask.py</code> trains RNNs on a rule reversal task using the base model, SCTT and DASC  <br>
* <code>main_rulereversaltask_cd.py</code> trains RNNs on a rule reversal task using CD  <br>

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
