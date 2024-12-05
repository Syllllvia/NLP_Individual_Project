import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    label_map = {"contradiction": 0, "entailment": 1, "neutral": 2, "-": -1}
    
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
    output_dir='./results_v24',             
    evaluation_strategy="epoch",        
    learning_rate=2e-5,                 
    per_device_train_batch_size=128,    
    per_device_eval_batch_size=128,     
    num_train_epochs=30,                
    weight_decay=0.01,                  
    logging_dir='./logs',               
    logging_steps=10,                   
    save_strategy="epoch",              
    load_best_model_at_end=True,        
    warmup_steps=500,                   
    lr_scheduler_type="cosine"          
)

loss_values = []

# FocalLoss 定义
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# 自定义 Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = FocalLoss(alpha=1, gamma=2)  # 调整alpha和gamma的值

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 使用 FocalLoss
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs):
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
trainer.save_model("./save_model_v24")
tokenizer.save_pretrained("./save_model_v24")
print("Evaluation Results:", eval_results)

plt.plot(loss_values, label="Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss Curve During Training")
plt.legend()
plt.show()
plt.savefig('./result_fig/loss_curve_v24.png')