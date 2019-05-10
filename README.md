## BERT for Natural Questions

This project is using pre-trained BERT model (uncased_L-12_H-768_A-12) to tackle Google's Natural Questions answering challenge. We start our exploration and implementation based on a BERT baseline model on the Natural Questions ([repo](https://github.com/google-research/language/tree/master/language/question_answering/bert_joint)) by Google AI Language.

### Install & Prerequisites

You need to have a GPU enabled TensorFlow installed. If not, you can easily do that in NYU Prince culster with Anaconda2 and python2. Below command create a conda virtual env with tensorflow-gpu and all required packages.
```
conda create --name tf_gpu python=2.7 tensorflow-gpu
```

After activated the conda virtual env. Need to install **bert** and **natural questions** related libraries.

```
pip install bert-tensorflow natural-questions
or
conda install bert-tensorflow natural-questions
```

Then, you need to install **gsutil** and download the pre-processed natural questions training dataset from Google.
```
gsutil cp -R gs://bert-nq/bert-joint-baseline .
```

This should give you the preprocessed training set, which will be used in our training.

```
bert-joint-baseline/nq-train.tfrecords-00000-of-00001
```
You also need to download the "tiny" dev set to verify/estimate the trained model performance.
```
gsutil cp -R gs://bert-nq/tiny-dev .
```

### Training
You can train the natural question model using below command. We have tried different implementations. So, you might select different run_nq_xxx.py  
1. Simple Linear Layer to adapt to QA problem/task domain: **run_nq_linear_layer**
2. Add Convolutional Abstraction Layers: **run_nq_conv**
3. Add BiLSTM(Bi-directional LSTM) and Attention Layers: **run_nq_BiLSTM**

--**train_precomputed_file**: where the pre-processed natural questions training dataset stores. You can use my download (/scratch/ywn202/ is public in NYU Prince)  
--**init_checkpoint**: where the pre-trained bert model used is stored. You can use my download (/scratch/ywn202/ is public in NYU Prince)     
--**output_dir**: where the model checkpoints will be stored     

```
srun --mem=15GB --gres=gpu:1 python -m run_nq_BiLSTM \
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
```


### Eval the Trained Model
With the "tiny" dev set you have previously downloaded. Run below command to get the predictions/answers.  

--**predict_file**: where the file contains tiny-dev examples we want to do predictions by the trained model    
--**init_checkpoint**: the trained model path. The file is too large to be included in the src/ for submission. If you don't want to try train our model, a trained BiLSTM-Attention model is stored in following location in NYU HPC: /scratch/ywn202/bert_model_output8/model.ckpt-123667 (the folder is public accessible)  
--**output_dir**: where the predicted answer will be saved to   

```
 srun --mem=15GB --gres=gpu:1 python -m run_nq_BiLSTM \
  --logtostderr \
  --bert_config_file=bert_config.json \
  --vocab_file=vocab-nq.txt \
  --predict_file=/scratch/ywn202/tiny-dev/nq-dev-sample.no-annot.jsonl.gz \
  --init_checkpoint=/scratch/ywn202/bert_model_output8/model.ckpt-123667 \
  --do_predict \
  --output_dir=/scratch/ywn202/bert_model_output8/ \
  --predict_batch_size 4
```

Calculate the accuracy and F1 numbers with below command.
natural_questions.nq_eval is the script included in natural_questions python lib we installed.

--**gold_path**: location of the ground truth of the tiny-dev set    
--**predictions_path**: location of the predictions/answers produced by your model

```
srun --mem=15GB --gres=gpu:1 python -m natural_questions.nq_eval \
  --logtostderr \
  --gold_path=/scratch/ywn202/tiny-dev/nq-dev-sample.jsonl.gz \
  --predictions_path=/scratch/ywn202/bert_model_output8/predictions.json
```

You should see some numbers similar as below:
```
{"short-best-threshold-f1": 0.544, "long-best-threshold-recall": 0.5048543689320388, "short-recall-at-precision>=0.75": 0.2, "short-precision-at-precision>=0.5": 0.5, "long-best-threshold-precision": 0.65, "long-recall-at-precision>=0.75": 0.36893203883495146, "long-best-threshold": 6.858288586139679, "short-best-threshold-recall": 0.4533333333333333, "long-recall-at-precision>=0.9": 0.02912621359223301, "short-best-threshold": 8.764707803726196, "short-recall-at-precision>=0.5": 0.5333333333333333, "long-recall-at-precision>=0.5": 0.5728155339805825, "short-precision-at-precision>=0.75": 0.75, "short-precision-at-precision>=0.9": 1.0, "long-precision-at-precision>=0.5": 0.5175438596491229, "long-precision-at-precision>=0.9": 1.0, "long-precision-at-precision>=0.75": 0.76, "short-recall-at-precision>=0.9": 0.04, "short-best-threshold-precision": 0.68, "long-best-threshold-f1": 0.5683060109289617}
```
