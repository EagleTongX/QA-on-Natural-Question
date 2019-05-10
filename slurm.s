#!/bin/bash
#
#SBATCH --job-name=bert_nq
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=6-23:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=15GB
#SBATCH --mail-type=END
#SBATCH --mail-user=ywn202@nyu.edu

python -m run_nq_BiLSTM \
  --logtostderr \
  --bert_config_file=bert_config.json \
  --vocab_file=vocab-nq.txt \
  --train_precomputed_file=/scratch/ywn202/bert-joint-baseline/nq-train.tfrecords-00000-of-00001 \
  --train_num_precomputed=494670 \
  --learning_rate=3e-5 \
  --num_train_epochs=1 \
  --max_seq_length=512 \
  --save_checkpoints_steps=5000 \
  --init_checkpoint=/scratch/ywn202/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --do_train \
  --output_dir=/scratch/ywn202/bert_model_output8 \
  --train_batch_size 4 \
  --num_train_epochs 1 
