# 🚀 AGNOSTOS: Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization

[[🌐 Project Page]](https://jiaming-zhou.github.io/AGNOSTOS/)  |  [[📄 Paper]](https://arxiv.org/pdf/2505.15660)  |  [🤗 Huggingface Data](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS)  |  [🤗 Huggingface Model](https://huggingface.co/Jiaming2472/X-ICM)  |  [[📺 Video]](https://youtu.be/5MKlijK1gKI)

[Jiaming Zhou](https://jiaming-zhou.github.io/)<sup>1</sup>, [Ke Ye](https://yipko.com/about/)<sup>1</sup>, [Jiayi Liu](https://www.jiayi-liu.cn/)<sup>1</sup>, [Teli Ma](https://teleema.github.io/)<sup>1</sup>, [Zifan Wang](https://zifanw.notion.site/)<sup>1</sup>, [Ronghe Qiu](https://github.com/ConnerQiu)<sup>1</sup>, [Kun-Yu Lin](https://kunyulin.github.io/)<sup>2</sup>, [Zhilin Zhao](https://lawliet-zzl.github.io/)<sup>3</sup>, [Junwei Liang](https://junweiliang.me/)<sup>1,4</sup>

<sup>1</sup>HKUST (Guangzhou), <sup>2</sup>HKU, <sup>3</sup>SYSU, <sup>4</sup>HKUST


## 📝 Overview
The project introduces AGNOSTOS, a simulation manipulation benchmark designed to rigorously evaluate cross-task zero-shot generalization of Vision-Language-Action models, and proposes Cross-Task In-Context Manipulation (X-ICM), a method that significantly improves cross-task generalization capabilities.

<img src="docs/agnostos_benchmark.gif" width="100%">

## 🔧 Environment Setup

### 🐳 Option 1: Using Docker (Recommended)
Please refer to [INSTALL_docker.md](./INSTALL_docker.md) to initialize your environment.

### ⚙️ Option 2: Manual Setup
Please refer to [INSTALL_manual.md](./INSTALL_manual.md) (adapted from [RoboPrompt](https://github.com/davidyyd/roboprompt)) for manual installation instructions.


## 📊 AGNOSTOS Benchmark Data
The benchmark consists of two parts (all data are available at [huggingface](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS)):
- 📚 18 seen tasks for training (140G in total, split into five files), links:

    [[seen_tasks.part_aa]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_aa?download=true) | [[seen_tasks.part_ab]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_ab?download=true) | [[seen_tasks.part_ac]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_ac?download=true) | [[seen_tasks.part_ad]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_ad?download=true) | [[seen_tasks.part_ae]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/unseen_tasks.tar?download=true)
- 🔍 23 unseen tasks for cross-task testing (20.2GB, one single file), link:

    [[unseen_tasks.tar]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/unseen_tasks.tar)

After downloading, process the files: 

```bash
### for seen task data, combine all five files
cat seen_tasks.part_* > seen_tasks.tar
### check the file, it should be "8217d78779acfd2873d0f55849c8efcc"
md5sum seen_tasks.tar 

tar -xvf seen_tasks.tar
tar -xvf unseen_tasks.tar
```

Creating symbolic links to the sub-folder `data`:
```bash
cd X-ICM
mkdir data
ln -s /path/to/seen_tasks data/
ln -s /path/to/unseen_tasks data/
```

## 🤖 X-ICM Method

### 1️⃣ Model Download
Download our pre-trained dynamics diffusion model from [[dynamics_diffusion.tar]](https://huggingface.co/Jiaming2472/X-ICM/resolve/main/dynamics_diffusion.tar?download=true) for cross-task in-context sample selection. 

After downloading, extract and create a symbolic link to the sub-folder `data`.
```bash
tar -xvf dynamics_diffusion.tar

cd X-ICM
ln -s /path/to/dynamics_diffusion data/
```

### 2️⃣ Evaluation
Run script `eval_XICM.sh` with the below parameters:
```bash
### set seed numbers for different runs
seeds: [example: "0,99"]
### set the number of rollouts for each run
episodes: [example: 25]
### set the method of LLM
modelname: [example: Qwen2.5.7B.instruct]
### set the number of cross-task in-context samples
num_icls: [example: 18]
### set the gpu list
gpu_ids: [example: 0,1]
### set the in-context sample selection method
ranking_method: [example: "lang_vis.out"]
```

For dynamics-guided in-context manipulation, run:
```bash
cd X-ICM
bash eval_scripts/eval_XICM.sh "0,99" 25 Qwen2.5.7B.instruct 18 0,1 "lang_vis.out"
```
**Reminder**: During evaluation, you need to load the Stable-Diffusion model and Qwen-LLM models from huggingface.
You can manually download them from huggingface and load them from the local paths in [`load_weight func`](./main.py#L132) and [`load_weight func`](./rlbench_inference_dynamics_diffusion.py#L136), accordingly.

For random selection of cross-task samples, run:
```bash
cd X-ICM
bash eval_scripts/eval_XICM.sh "0,99" 25 Qwen2.5.7B.instruct 18 0,1 "random"
```

After testing, you can use [`gather_score.py`](./gather_score.py) to collect and analyze the results.

## 🎯 Benchmarking Results over all 23 unseen tasks
We provide the testing results of our `X-ICM (7B)` and `X-ICM (72B)` models under the sub-folder `logs/`.
- X-ICM (7B) achieves 23.5% average success rate and X-ICM (72B) achieves 30.1% average success rate, both versions outperform all existing VLA models;
- X-ICM (7B) fails on only two tasks, while X-ICM (72B) succeeds on all tasks;

## 🔬 Benchmarking Your VLA Model on AGNOSTOS

### 1️⃣ Fine-tuning
Due to the embodiment gap, existing VLA models need to be fine-tuned on RLBench data. 

Please follow your VLA model's fine-tuning guidelines to fine-tune your models on our 18 seen tasks, and then test the models on our 23 unseen tasks.

### 2️⃣ Testing Fine-tuned VLA Models
Modify the [`custom_agent.py`](./custom_agent.py) file:
1. Load your VLA model in the [`load_weights`](scripts/custom_agent.py#L116) function
2. Implement VLA model inference in the [`_inference`]((scripts/custom_agent.py#L21)) function, including input construction and output format conversion (link)
3. Run the evaluation:
    ```bash
    bash eval_CustomModel.sh seeds episodes gpu_ids
    ```

    Example:
    ```bash
    bash eval_scripts/eval_CustomModel.sh "0,99" 25 0,1
    ```

💡 Note: Different VLA models may require different input image sizes (default is 256x256). 

💡 Please modify [`IMAGE_SIZE`](./main_custom.py#L32) in [`main_custom.py`](./main_custom.py) accordingly.



## 🙏 Acknowledgments
This repository is built upon the [RoboPrompt](https://github.com/davidyyd/roboprompt). Some resources from [RVT](https://github.com/NVlabs/RVT) and [RLBench](https://github.com/stepjam/RLBench) are used in this work.

## 📄 Citation
If you find our work helpful to your research, please kindly give us a star and cite our paper.
```bibtex
@article{zhou2025exploring,
    title={Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization},
    author={Zhou, Jiaming and Ye, Ke and Liu, Jiayi and Ma, Teli and Wang, Zifang and Qiu, Ronghe and Lin, Kun-Yu and Zhao, Zhilin and Liang, Junwei},
    journal={arXiv preprint arXiv:2505.15660},
    year={2025}
}
```
