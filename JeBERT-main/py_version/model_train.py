from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer

drive_path = '/content/drive/MyDrive/project3/'

config_encoder = BertConfig()
config_decoder = BertConfig()

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

model = EncoderDecoderModel(config=config)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

config_encoder = model.config.encoder
config_decoder = model.config.decoder

config_encoder.bos_token_id = tokenizer.cls_token_id
config_encoder.eos_token_id = tokenizer.sep_token_id
config_encoder.decoder_start_token_id = tokenizer.cls_token_id
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
config_decoder.bos_token_id = tokenizer.cls_token_id
config_decoder.eos_token_id = tokenizer.sep_token_id
config_decoder.decoder_start_token_id = tokenizer.cls_token_id

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model.cuda()

collator = DataCollatorForSeq2Seq(tokenizer, model)

arguments = Seq2SeqTrainingArguments(
    output_dir= drive_path + 'dump',
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    gradient_accumulation_steps=1,
    save_total_limit=5,
    dataloader_num_workers=1,
    fp16=True,
    load_best_model_at_end=True
)

trainer = Trainer(
    model,
    arguments,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

trainer.train()

model.save_pretrained(drive_path + 'dump')
