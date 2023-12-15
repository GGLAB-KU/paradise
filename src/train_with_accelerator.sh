#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4    
#SBATCH --partition=mid        
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=12:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=60GB

accelerate config

accelerate test

accelerate launch train_with_accelerator.py \
  --model_name_or_path distilbert-base-uncased \
  --train_file data/tip_train.csv \
  --validation_file data/tip_dev.csv \
  --seed 42 \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir model_zoo/distilbert-tip-42 \
