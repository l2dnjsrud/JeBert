import unicodedata
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import json

from transformers import (
    EncoderDecoderModel,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
    BertConfig,
    DistilBertConfig,
    EncoderDecoderConfig
)

import torch
from torch.utils.data import Dataset, DataLoader

drive_path = '/content/drive/MyDrive/project3/'

data_dev = []
with open(drive_path + 'jit/je.dev', 'r', encoding = "UTF-8") as fd:
    for line in fd.readlines():
        data_dev.append(line[:-1])

data_test = []
with open(drive_path + 'jit/je.test', 'r', encoding = "UTF-8") as fd:
    for line in fd.readlines():
        data_test.append(line[:-1])

data_train = []
with open(drive_path + 'jit/je.train', 'r', encoding = "UTF-8") as fd:
    for line in fd.readlines():
        data_train.append(line[:-1])

two_dot = data_dev[21][39]
data = data_dev + data_test + data_train
changed_data = []
dot_num = 0
two_dot_num = 0

for line in data:
    dot_num += line.count(dot)
    two_dot_num += line.count(two_dot)
    newline = line.replace(dot,'ㅗ')
    changed_data.append(unicodedata.normalize('NFKC',newline.replace(two_dot,'ㅛ')))

with open(drive_path + 'jit/je_trans.dev', 'w') as fd:
    for line in changed_data[:5000]:
        fd.write(line+'\n')

with open(drive_path + 'jit/je_trans.test', 'w') as fd:
    for line in changed_data[5000:10000]:
        fd.write(line+'\n')

with open(drive_path + 'jit/je_trans.train', 'w') as fd:
    for line in changed_data[10000:]:
        fd.write(line+'\n')

limit_alphabet = 4000
vocab_size = 32000

tokenizer = BertWordPieceTokenizer(
    vocab=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False, # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##"
)
files = [drive_path + 'jit/je.dev', drive_path + 'jit/je.train', drive_path + 'jit/je.test']
tokenizer.train(
    files=files,
    limit_alphabet=limit_alphabet,
    vocab_size=vocab_size
)

tokenizer.save(drive_path + 'dump/ch-{}-je-{}-pretty'.format(limit_alphabet, vocab_size),True)

files = [drive_path + 'jit/je_trans.dev', drive_path + 'jit/je_trans.train', drive_path + 'jit/je_trans.test']
tokenizer.train(
    files=files,
    limit_alphabet=limit_alphabet,
    vocab_size=vocab_size
)
tokenizer.save(drive_path + 'dump/ch-{}-je_trans-{}-pretty'.format(limit_alphabet, vocab_size),True)

vocab_path = [drive_path + 'dump/ch-4000-je-32000-pretty', drive_path + 'dump/ch-4000-je_trans-32000-pretty']
vocab_file = [drive_path + 'dump/wpm-vocab-je.txt', drive_path + 'dump/wpm-vocab-je_trans.txt']

for i in range(len(vocab_path)):
    f = open(vocab_file[i],'w',encoding='utf-8')
    with open(vocab_path[i]) as json_file:
        json_data = json.load(json_file)
        for item in json_data['model']['vocab'].keys():
            f.write(item+'\n')

        f.close()

src_tokenizer = BertTokenizer(drive_path + 'dump/wpm-vocab-je_trans.txt', do_lower_case=False)
trg_tokenizer = BertTokenizer(drive_path + 'dump/wpm-vocab-je.txt', do_lower_case=False

def tokenized_data(src_tokenizer, trg_tokenizer, src_file_path, trg_file_path):
    src_data = []
    trg_data = []
    with open(src_file_path, 'r') as fd:
        for line in fd.readlines():
            src_data.append(line[:-1])
    with open(trg_file_path, 'r') as fd:
        for line in fd.readlines():
            trg_data.append(line[:-1])

    embeddings = []
    for i in range(len(src_data)):
        embeddings.append((src_tokenizer(src_data[i], truncation=True, max_length=50, return_token_type_ids=False, padding='max_length'), \
                          trg_tokenizer(trg_data[i], truncation=True, max_length=50, return_token_type_ids=False, padding='max_length')))

    return embeddings

train_tokenized_data = tokenized_data(src_tokenizer, trg_tokenizer, drive_path + 'jit/je_trans.train', drive_path + 'jit/je.train')
dev_tokenized_data = tokenized_data(src_tokenizer, trg_tokenizer, drive_path + 'jit/je_trans.dev', drive_path + 'jit/je.dev')
test_tokenized_data = tokenized_data(src_tokenizer, trg_tokenizer, drive_path + 'jit/je_trans.test', drive_path + 'jit/je.test')

class DatasetRetriever(Dataset):
    def __init__(self, features):
        super(DatasetRetriever, self).__init__()
        self.features = features
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):   
        feature = self.features[index]
        return {
            'input_ids':torch.tensor(feature[0]['input_ids'] ,dtype=torch.long),
            'labels':torch.tensor(feature[1]['input_ids'] ,dtype=torch.long)
        }

train_dataset = DatasetRetriever(train_tokenized_data)
dev_dataset = DatasetRetriever(dev_tokenized_data)
test_dataset = DatasetRetriever(test_tokenized_data)

# 모델
config = BertConfig()
config.vocab_size = vocab_size

config_encoder = config
config_decoder = config

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

# Initializing a Bert2Bert model from the bert-base-uncased style configurations
model = EncoderDecoderModel(config=config)

# Accessing the model configuration
config_encoder = model.config.encoder
config_decoder = model.config.decoder

# set decoder config to causal lm
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True

# Saving the model, including its configuration
model.save_pretrained('my-model')

# loading model and config from pretrained folder
encoder_decoder_config = EncoderDecoderConfig.from_pretrained('my-model')
model = EncoderDecoderModel.from_pretrained('my-model', config=encoder_decoder_config)

model.config.decoder_start_token_id = trg_tokenizer.cls_token_id
model.config.pad_token_id = trg_tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

collator = DataCollatorForSeq2Seq(src_tokenizer, model)

arguments = Seq2SeqTrainingArguments(
    output_dir = 'dump',
    do_train = True, 
    do_eval = True,
    evaluation_strategy= 'epoch',
    save_strategy='epoch',
    num_train_epochs = 10,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32,
    warmup_ratio = 0.1,
    gradient_accumulation_steps = 1,
    save_total_limit=5,
    dataloader_num_workers= 1,
    load_best_model_at_end = True
)

trainer = Trainer(
    model, 
    arguments,
    data_collator = collator,
    train_dataset = train_dataset,
    eval_dataset = dev_dataset
)

trainer.train()

model.save_pretrained(drive_path + 'dump')
