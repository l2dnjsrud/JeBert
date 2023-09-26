import json
import os
import pandas as pd

file_dir = '/content/drive/MyDrive/project3/AI_HUB_data/한국어 방언 발화(제주도)/Training/[라벨]제주도_학습용데이터_1.zip (Unzipped Files)/'
filelist = os.listdir('/content/drive/MyDrive/project3/AI_HUB_data/한국어 방언 발화(제주도)/Training/[라벨]제주도_학습용데이터_1.zip (Unzipped Files)')

standard_form = []
dialect_form=[]

# AI HUB 데이터는 json과 txt 파일이 존재하며, 예외처리를 통해 데이터 로드
for file in filelist:
    with open(file_dir + file,'r') as f:
        try:
            data = json.load(f)
            standard_form += [data['utterance'][i]['standard_form'] for i in range(len(data['utterance']))]
            dialect_form += [data['utterance'][i]['dialect_form'] for i in range(len(data['utterance']))]
        except json.JSONDecodeError:
            continue

jejudata = pd.DataFrame()

jejudata['standard_form'] = pd.DataFrame(standard_form)
jejudata['dialect_form'] = pd.DataFrame(dialect_form)

jejudata.to_csv('/content/drive/MyDrive/project3/AI_HUB_data/ai_hub_train.csv')

v_file_dir = '/content/drive/MyDrive/project3/AI_HUB_data/한국어 방언 발화(제주도)/Validation/[라벨]제주도_학습용데이터_3.zip (Unzipped Files)/'
valid_filelist = os.listdir('/content/drive/MyDrive/project3/AI_HUB_data/한국어 방언 발화(제주도)/Validation/[라벨]제주도_학습용데이터_3.zip (Unzipped Files)')

valid_standard_form = []
valid_dialect_form=[]

for file in v_filelist:
    with open(v_file_dir + file, 'r') as f:
        try:
            data = json.load(f)
            valid_standard_form += [data['utterance'][i]['standard_form'] for i in range(len(data['utterance']))]
            valid_dialect_form += [data['utterance'][i]['dialect_form'] for i in range(len(data['utterance']))]
        except json.JSONDecodeError:
            continue

valid_jejudata = pd.DataFrame()

valid_jejudata['standard_form'] = pd.DataFrame(valid_standard_form)
valid_jejudata['dialect_form'] = pd.DataFrame(valid_dialect_form)

valid_jejudata.to_csv('/content/drive/MyDrive/project3/AI_HUB_data/ai_hub_valid.csv')
