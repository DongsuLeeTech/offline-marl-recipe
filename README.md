<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>A Recipe for Stable Offline Multi-agent Reinforcement Learning</h1>
      <br>
    </summary>
  </ul>
</div>
</div>

## Overview
This codebase implements a unified offline MARL framework for studying the interaction between value decomposition, value learning, and policy extraction. It provides modular implementations of centralized, linear (VDN), and non-linear (Mixer) value decomposition, combined with multiple offline value learning objectives (TD, SARSA, IQL) and policy extraction methods (AWR, BRAC).

## Installation
```bash
# Environment (OG-MARL: https://github.com/instadeepai/og-marl)
conda create -n recipe python=3.8
conda activate recipe
python -m pip install --upgrade pip
pip install -r requirements/datasets.txt
```

Install the SMAC, MPE, MA-MuJoCo environment you plan to use:

```bash
# SMAC v1
bash install_environments/smacv1.sh
pip install -r install_environments/requirements/smacv1.txt

# SMAC v2
bash install_environments/smacv2.sh
pip install -r install_environments/requirements/smacv2.txt

# MPE
pip install -r install_environments/requirements/pettingzoo.txt

# MA-MuJoCo
bash install_environments/mujoco200.sh
pip install -r install_environments/requirements/mamujoco200.txt
```

If you need baseline TensorFlow dependencies:

```bash
pip install -r requirements/baselines.txt
```

**We distribute environment.yml file for conda environment setup. You can create the conda environment using this.**

## Quick start

```bash
# TD3BC
python main_td3bc.py --env smac_v1 --source og_marl --scenario 2s3z --dataset Good --seed 0 --agent.alpha 1.0 --agent.q sarsa --agent.decompose_q mixer --agent.actor_loss brac  --agent.mixer_q svn --discrete True
```
