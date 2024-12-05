import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# load dataset
def load_jsonl_data(file_path):
    data = pd.read_json(file_path, lines=True)
    return data

matched_data = load_jsonl_data('/home/mfuai/MSBD5018/dev_matched_sampled-1.jsonl')
mismatched_data = load_jsonl_data('/home/mfuai/MSBD5018/dev_mismatched_sampled-1.jsonl')
all_data = pd.concat([matched_data, mismatched_data])

# remove rows with invalid labels
all_data = all_data[all_data['gold_label'].isin(['contradiction', 'entailment', 'neutral'])]
# trans to huggingface dataset
dataset = Dataset.from_pandas(all_data)

# load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

# pre-processing: token&model
def preprocess_function(examples):
    label_map = {"contradiction": 0, "entailment": 1, "neutral": 2, "-": -1}  # 如果遇到未知标签，可以设置为 -1，表示无效
    
    encoding = tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        truncation=True,
        padding='max_length',
        max_length=512
    )
    
    labels = [label_map.get(label, -1) for label in examples['gold_label']]
    encoding['labels'] = labels

    return encoding

encoded_dataset = dataset.map(preprocess_function, batched=True)

# train & test dataset
train_size = int(0.8 * len(encoded_dataset))
train_dataset = encoded_dataset.select(range(train_size))
eval_dataset = encoded_dataset.select(range(train_size, len(encoded_dataset)))

# fine-tuning
training_args = TrainingArguments(
    output_dir='./results1',             # 输出文件夹
    evaluation_strategy="epoch",        # 每个epoch后进行评估
    learning_rate=2e-5,                 # 学习率
    per_device_train_batch_size=64,    # 每个设备的训练批大小
    per_device_eval_batch_size=64,     # 每个设备的验证批大小
    num_train_epochs=30,                # 训练epoch数
    weight_decay=0.01,                  # 权重衰减
    logging_dir='./logs',               # 日志文件夹
    logging_steps=10,                   # 每10步记录日志
    save_strategy="epoch",              # 每个epoch保存模型
    load_best_model_at_end=True,        # 在训练结束时加载最好的模型
    warmup_steps=500,                   # 学习率热身的步数
    lr_scheduler_type="cosine"          # 指定学习率调度器类型，默认是线性调度（linear）
)

loss_values = []

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 如果 labels 存在，进行计算
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # 使用交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    def log(self, logs):
        # 每次记录日志时，将loss添加到loss_values中
        if 'loss' in logs:
            loss_values.append(logs['loss'])
        super().log(logs)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# evaluation

eval_results = trainer.evaluate()
trainer.save_model("./save_model1")
tokenizer.save_pretrained("./save_model1")
print("Evaluation Results:", eval_results)

plt.plot(loss_values, label="Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss Curve During Training")
plt.legend()
plt.show()
plt.savefig('./result_fig/loss_curve1.png') 