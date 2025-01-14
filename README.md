# To-the-Max
[![DOI](https://zenodo.org/badge/802934201.svg)](https://doi.org/10.5281/zenodo.14645336)

Code accompanying our ICML 2024 paper "[To the Max: Reinventing Reward in Reinforcement Learning](https://arxiv.org/abs/2402.01361)".


### Installation [beta]
Install the conda environment by running

```conda env create -f environment.yml```

### Usage
All RL components are implemented in the `src/` folder.
All the logging is done using [Wandb](https://wandb.ai/site).
The experiments in the paper can be replicated by running 
the python scripts.

Scripts `EXP0_*` are used to tune the discrete shortest path reward value
using the cumulative RL algorithms. We check different `betta` and whether
to make the reward negative by subtracting 1.

Scripts `EXP1_*` perform hyperparameter tuning.

Scripts `EXP2_*` run the experiments for the single-goal maze using DSP reward.
`EXP2_extra` run experiments with dense `l2` and sparse rewards.
Similarly, `EXP3_*` run experiments on the two-goals maze.

The experiments on the Fetch domain are run in `EXP4`.

Experiments using the slippery version of the single-goal maze are run in `EXP5`.
