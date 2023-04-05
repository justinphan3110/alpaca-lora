#!/bin/bash

sbatch --nodes=2 --gpus-per-node=2 --time=48:00:00 lora_3B.sh