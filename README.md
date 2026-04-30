# NVIDIA Nemotron Reasoning Challenge

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/kaggle-nvidia-nemotron-model-reasoning-challenge/blob/main/nvidia-nemotron-sft-training.ipynb)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

**Competition:** [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview)  
**Base model:** Nemotron-3-Nano-30B-A3B-BF16  
**Task:** Fine-tune a LoRA adapter (rank 32) on logic puzzles so the model outputs answers in `\boxed{answer}` format

---

## Approach

The competition baseline submits an untrained LoRA adapter. This project trains the adapter using supervised fine-tuning (SFT) on 8,569 chain-of-thought reasoning trajectories. Each training example pairs a puzzle prompt with a step-by-step reasoning chain and the correct answer.

Training format:
```
System: <task-specific instruction>
User: <puzzle prompt>
Assistant: <reasoning chain>

The answer is \boxed{answer}.
```

Task types covered: Roman numerals, unit conversion, gravity/physics, text ciphers, bit manipulation, symbol transformations.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Nemotron-3-Nano-30B-A3B-BF16 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | in_proj, out_proj, up_proj, down_proj |
| Training examples | 8,569 |
| Epochs | 2 |
| Max sequence length | 1024 |
| Learning rate | 2e-4 |
| LR schedule | Cosine |
| Optimizer | AdamW (fused) |
| Precision | BF16 |
| Hardware | NVIDIA RTX Pro 6000 (102 GB VRAM) |

---

## Results

| Submission | Public Score |
|------------|-------------|
| Untrained baseline | 0.50 |
| SFT 1 epoch, 512 tokens | 0.59 |
| SFT 2 epochs, 1024 tokens | pending |

---

## Files

| File | Description |
|------|-------------|
| `nvidia-nemotron-sft-training.ipynb` | SFT training notebook |
| `nvidia-nemotron-submission-demo.ipynb` | Original competition baseline (untrained adapter) |
| `CoderGym/Nemotron/train_reasoning_v5.jsonl` | 8,569 training examples with reasoning chains |
| `CoderGym/Nemotron/train.csv` | Raw competition training data |
| `CoderGym/Nemotron/compare_lora_before_after_v2.py` | Evaluation script using NVIDIA API |

---

## References

- [Competition overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview)
- [Nemotron model on Kaggle](https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16)
- [LoRA paper (arXiv:2106.09685)](https://arxiv.org/abs/2106.09685)
- [CoderGym repo](https://github.com/lkk688/CoderGym/tree/main/Nemotron)
