# Transformer-based-memory-for-visual-navigation
PPO version code for RAL 2023 paper [Transformer Memory for Interactive Visual Navigation in Cluttered Environments](https://www.hrl.uni-bonn.de/teaching/ss23/master-seminar/transformer-memory-for-interactive-visual-navigation-in-cluttered-environments.pdf).

Transformer belief state encoder for encoding history information
plus PPO algorithm to learn the policy.

The vector env is designed for IGibson and is also easy to reimplement for other environments such as Habitat.

## Training scritps
```
  python /config/scripts/train.py
```

## Citation

```bibtex
@article{li2023transformer,
  title={Transformer Memory for Interactive Visual Navigation in Cluttered Environments},
  author={Li, Weiyuan and Hong, Ruoxin and Shen, Jiwei and Yuan, Liang and Lu, Yue},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={3},
  pages={1731--1738},
  year={2023},
  publisher={IEEE}
}
```
