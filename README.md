# Do MLLMs Really See It: Reinforcing Visual Attention in Multimodal LLMs
<h3 align="center"><a href="https://cratileo.github.io/Sayo-Pages/"> Project page here 🚀</a></h3>
<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2602.08241-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2602.08241)
[![Hugging Face Collection](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface&logoColor=000)](https://huggingface.co/Craleo/Sayo-Qwen-8B)
<br>
</h5>
<div align=center><img src=assets/main_fig.png width="75%" height="75%"></div>

> **It's the official repository of "Do MLLMs Really See It: Reinforcing Visual Attention in Multimodal LLMs".**
> SAYO is a model trained via only visual attention based reward. 

---

## Setup
To set up your environment for training:
```bash
cd Sayo
conda create -n sayo -python=3.12
conda activate sayo
pip install -r requirements.txt
```

## Use Guide
To use the pre-trained models for evaluation, follow the steps below:


## Citation
If you find our work helpful, please cite:

```bibtex
@article{domllmsreallyseeit,
      title={Do MLLMs Really See It: Reinforcing Visual Attention in Multimodal LLMs}, 
      author={Siqu Ou and Tianrui Wan and Zhiyuan Zhao and Junyu Gao and Xuelong Li},
      year={2026},
      eprint={2602.08241},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.08241}, 
}
```