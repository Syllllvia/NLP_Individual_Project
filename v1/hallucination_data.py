from datasets import load_dataset

# 加载wikibio-gpt3-hallucination数据集
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")

# 查看数据集的基本信息
print(dataset)

# 选择train数据集
eval_dataset = dataset['evaluation']

# 查看数据集中的一个样本
sample =eval_dataset[0]
print("GPT-3 Generated Sentence:", sample['gpt3_sentences'])
print("Wikipedia Bio Text:", sample['wiki_bio_text'])
print("Human Annotation:", sample['annotation'])
