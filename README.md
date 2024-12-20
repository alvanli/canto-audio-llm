## Canto-DiVA
- this is an attempt in replicating the [DiVA](https://arxiv.org/abs/2410.02678) paper and finetune it to Cantonese audio data
  - DiVA provides an easy way to create Audio models without generating a bunch of tasks for audio data, and without instruction data
  - the original [code](https://github.com/Helw150/levanter/blob/will/distill/src/levanter/models/via.py) was written in Levanter, so I wanted to turn it into PyTorch
- The scripts are run on local Ubuntu machine with 2 x 4090s
- The model has been training since Dec 19, 2024. When completed, results will be uploaded 