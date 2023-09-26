from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pickle

drive_path = '/content/drive/MyDrive/project3/'
vocab_size = 30522

tokenizer = BertTokenizer(drive_path + f'dump/wpm-vocab-extend-{vocab_size}.txt', do_lower_case=False)

def make_corpus(src_file_path, trg_file_path, file_type: str):
    data = []
    if file_type == 'jit':
        with open(src_file_path, 'r') as fd:
                for line in fd.readlines():
                    data.append([line[:-1]])

        with open(trg_file_path, 'r') as fd:
            for i, line in enumerate(fd.readlines()):
                data[i].append(line[:-1])

    elif file_type == 'pickled_ai_hub':
        with open(src_file_path, 'rb') as f:
            tmp_src = pickle.load(f)

        with open(trg_file_path, 'rb') as f:
            tmp_trg = pickle.load(f)
            
        for i in range(len(tmp_src)):
            data.append([tmp_src[i], tmp_trg[i]])
    return data

def tokenized_data(tokenizer, data, max_length=128, stride=30):
    cnt = 0
    embeddings = []
    for src, trg in data:
        src_sample = tokenizer(src, truncation=True, max_length=max_length, stride=stride, return_token_type_ids=False, return_attention_mask=False, return_overflowing_tokens=True)
        trg_sample = tokenizer(trg, truncation=True, max_length=max_length, stride=stride, return_token_type_ids=False, return_attention_mask=False, return_overflowing_tokens=True)
        embeddings.append({'input_ids' : src_sample['input_ids'],
                           'labels': trg_sample['input_ids']})
        if src_sample['num_truncated_tokens'] > 0 and trg_sample['num_truncated_tokens'] > 0:
            src_tmp = src_sample['overflowing_tokens']
            trg_tmp = trg_sample['overflowing_tokens']
            while len(src_tmp) > 0 and len(trg_tmp) > 0:
                cnt += 1
                src_input = [tokenizer.cls_token_id]
                trg_input = [tokenizer.cls_token_id]
                src_input.extend(src_tmp[:max_length-2])
                src_input.append(tokenizer.sep_token_id)
                trg_input.extend(trg_tmp[:max_length-2])
                trg_input.append(tokenizer.sep_token_id)                
                embeddings.append({'input_ids' : src_input,
                               'labels': trg_input})
                src_tmp = src_tmp[max_length-stride-2:2*max_length-stride-2]
                trg_tmp = trg_tmp[max_length-stride-2:2*max_length-stride-2]
    print(f'Processed {cnt} amount of overflowing token set!')

    cnt = 0
    for item in embeddings:
        if len(item['input_ids']) == 2 or len(item['labels']) == 2:
            embeddings.remove(item)
            cnt += 1
    print(f'Removed {cnt} amount of empty token set!')
    return embeddings

class DatasetRetriever(Dataset):
    def __init__(self, features):
        super(DatasetRetriever, self).__init__()
        self.features = features

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):   
        feature = self.features[index]
        return {
            'input_ids':torch.tensor(feature['input_ids'] ,dtype=torch.long),
            'labels':torch.tensor(feature['labels'] ,dtype=torch.long)
        }
