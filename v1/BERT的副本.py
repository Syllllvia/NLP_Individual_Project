import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# check cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# load dataset
def load_jsonl_data(file_path):
    data = pd.read_json(file_path, lines=True)
    return data

matched_data = load_jsonl_data('/home/mfuai/MSBD5018/dev_matched_sampled-1.jsonl')
mismatched_data = load_jsonl_data('/home/mfuai/MSBD5018/dev_mismatched_sampled-1.jsonl')
all_data = pd.concat([matched_data, mismatched_data])

# trans to huggingface dataset
dataset = Dataset.from_pandas(all_data)

# load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

# pre-processing: token&model
def preprocess_function(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# train & test dataset
train_size = int(0.8 * len(encoded_dataset))
train_dataset = encoded_dataset.select(range(train_size))
eval_dataset = encoded_dataset.select(range(train_size, len(encoded_dataset)))

# fine-tuning
training_args = TrainingArguments(
    output_dir='./results',           # 输出文件夹
    evaluation_strategy="epoch",      # 每个epoch后进行评估
    learning_rate=2e-5,               # 学习率
    per_device_train_batch_size=8,    # 每个设备的训练批大小
    per_device_eval_batch_size=8,     # 每个设备的验证批大小
    num_train_epochs=3,               # 训练epoch数
    weight_decay=0.01,                # 权重衰减
    logging_dir='./logs',             # 日志文件夹
    logging_steps=10,                 # 每10步记录日志
    save_strategy="epoch",            # 每个epoch保存模型
    load_best_model_at_end=True       # 在训练结束时加载最好的模型
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# evaluation
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
