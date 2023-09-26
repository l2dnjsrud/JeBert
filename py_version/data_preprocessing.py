import pickle

drive_path = '/content/drive/MyDrive/project3/'

# Train Korea to Jeju
ai_hub_data = make_corpus(drive_path + 'PREPROCESSED_AI_HUB_data/ai_hub_standard_v2.pkl', drive_path + 'PREPROCESSED_AI_HUB_data/ai_hub_trans_v2.pkl', 'pickled_ai_hub')

train_ai_hub = ai_hub_data[:-10000]
dev_ai_hub = ai_hub_data[-10000:-5000]
test_ai_hub = ai_hub_data[-5000:]

train_data = make_corpus(drive_path + 'jit/ko.train', drive_path + 'jit/je.train', 'jit')
dev_data = make_corpus(drive_path + 'jit/ko.dev', drive_path + 'jit/je.dev', 'jit')
test_data = make_corpus(drive_path + 'jit/ko.test', drive_path + 'jit/je.test', 'jit')

# Train Jeju to Korea
''' 
ai_hub_data = make_corpus(drive_path + 'PREPROCESSED_AI_HUB_data/ai_hub_trans_v2.pkl', drive_path + 'PREPROCESSED_AI_HUB_data/ai_hub_standard_v2.pkl', 'pickled_ai_hub')

train_ai_hub = ai_hub_data[:-10000]
dev_ai_hub = ai_hub_data[-10000:-5000]
test_ai_hub = ai_hub_data[-5000:]

train_data = make_corpus(drive_path + 'jit/je.train', drive_path + 'jit/ko.train', 'jit')
dev_data = make_corpus(drive_path + 'jit/je.dev', drive_path + 'jit/ko.dev', 'jit')
test_data = make_corpus(drive_path + 'jit/je.test', drive_path + 'jit/ko.test', 'jit')
'''

train_data = train_data + train_ai_hub
dev_data = dev_data + dev_ai_hub
test_data = test_data + test_ai_hub

train_tokenized_data = tokenized_data(tokenizer, train_data)
dev_tokenized_data = tokenized_data(tokenizer, dev_data)
test_tokenized_data = tokenized_data(tokenizer, test_data)

train_dataset = DatasetRetriever(train_tokenized_data)
dev_dataset = DatasetRetriever(dev_tokenized_data)
test_dataset = DatasetRetriever(test_tokenized_data)
