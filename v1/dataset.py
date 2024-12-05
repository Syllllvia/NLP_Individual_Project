
import pandas as pd
# load dataset
def load_jsonl_data(file_path):
    data = pd.read_json(file_path, lines=True)
    return data

matched_data = load_jsonl_data('/home/mfuai/MSBD5018/dev_matched_sampled-1.jsonl')
mismatched_data = load_jsonl_data('/home/mfuai/MSBD5018/dev_mismatched_sampled-1.jsonl')
all_data = pd.concat([matched_data, mismatched_data])

print(all_data.head(5))
print(all_data.columns)
