> **Historical snapshot.**  
> This document records an earlier research stage and is not the current result.  
> **历史快照：本文档记录早期研究阶段，不代表当前研究结论。**

# Project Compute Dashboard

## Pipeline Status

- language: zh
- stage: cloud-b500-preparation
- status: READY_FOR_LOCAL_VALIDATION_ONLY
- paid_instance_started: false
- local_gpu_training: frozen

## Remote Server

- gpu: remote
- provider: AutoDL preferred; RunPod Secure Cloud fallback
- SSH: pending user-created instance
- GPU: 1x NVIDIA RTX 4090 24GB
- CPU: at least 8 vCPU
- RAM: at least 32 GiB
- Data disk: at least 150 GiB, repository under `/root/autodl-tmp/`
- Base image: PyTorch 2.8.0, Python 3.12, CUDA 12.8
- Environment spec: `configs/b500_cloud_env_v1.json`
- Code dir: `/root/autodl-tmp/error-guided-sft-data-selection/`
- code_sync: git
- branch: `research/cloud-b500-v1`
- wandb: false
- paid launch requires explicit user confirmation

## Experiment Contract

- Matrix: `configs/b500_formal_matrix_cloud_4090_v1.json`
- Run policy: exactly one job per manual invocation
- Main result: all 9 strategy-by-seed cells on the same cloud instance
- Local RTX 5060 Laptop results remain secondary engineering evidence
- Do not change selectors, thresholds, model, data, LoRA recipe, prompt, parser policy, or seeds
- Do not launch B=1000 or a 4B confirmation without a separate decision
