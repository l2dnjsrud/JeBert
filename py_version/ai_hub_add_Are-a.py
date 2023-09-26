from transformers import BertTokenizer, EncoderDecoderModel
import pandas as pd
from tqdm import tqdm
import pickle

drive_path = '/content/drive/MyDrive/project3/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_train = pd.read_csv(drive_path + 'PREPROCESSED_AI_HUB_data/Train/preprocessed_ai_hub_train.csv')
df_train.dropna(inplace=True)

df_dev = pd.read_csv(drive_path + 'PREPROCESSED_AI_HUB_data/Val/preprocessed_ai_hub_val.csv')
df_dev.dropna(inplace=True)

# AI Hub 데이터는 dialect와 standard가 동일한 경우가 있으므로, 두 가지가 서로 다른 경우만 사용할 수 있도록 선별함
condition = df_train['dialect'] != df_train['standard']
df_train = df_train[condition]

df_train_standard = df_train['standard']
df_train_dialect = df_train['dialect']

test_df_train_standard = list(df_train_standard[:200000])
test_df_train_dialect = list(df_train_dialect[:200000])

# 아래아 추가하기(v1)
src_tokenizer = BertTokenizer(drive_path + 'dump/wpm-vocab-je_trans.txt', do_lower_case=False)
trg_tokenizer = BertTokenizer(drive_path + 'dump/wpm-vocab-je.txt', do_lower_case=False)

model = EncoderDecoderModel.from_pretrained(drive_path + 'dump/model1').to(device)

# 아래아 추가하기(v2)
'''
model = EncoderDecoderModel.from_pretrained(drive_path + 'dump/model1').to(device)
'''

model.config.decoder_start_token_id = trg_tokenizer.cls_token_id
model.config.pad_token_id = trg_tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.eval()

def get_prediction(text):
    embeddings = src_tokenizer(text, return_attention_mask = False, return_token_type_ids= False, return_tensors = 'pt').to(device)
    output = model.generate(**embeddings, max_length = 256, eos_token_id=3)[0, 1:-1].to(device)
    return trg_tokenizer.decode(output[1:])

ai_test_pred = []

for i in tqdm(range(len(test_df_train_dialect))):
    src = test_df_train_dialect[i]
    output = get_prediction(src)
    ai_test_pred.append(output)

# 데이터 저장하기
with open('PREPROCESSED_AI_HUB_data/ai_hub_trans.pkl', 'wb') as f:
    pickle.dump(ai_test_pred, f)

with open('PREPROCESSED_AI_HUB_data/ai_hub_standard.pkl', 'wb') as f:
    pickle.dump(test_df_train_standard, f)
