#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, AutoConfig, GPT2LMHeadModel
import csv
from transformers import TrainerCallback

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_log.csv"):
        self.log_file = log_file
        with open(self.log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Training Loss", "Validation Loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            train_loss = logs.get("loss", "N/A")
            val_loss = logs.get("eval_loss", "N/A")
            with open(self.log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, train_loss, val_loss])

# Data load
ds_train = load_dataset("csv", data_files="/home/pkar/train_eos.csv", split="train")
ds_valid = load_dataset("csv", data_files="/home/pkar/val_eos.csv", split="train")

raw_datasets = DatasetDict({
    "train": ds_train,
    "valid": ds_valid,
})

# Creating a tokenizer from scratch
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=30_000, min_frequency=2, special_tokens=[
    "<pad>", "<unk>", "<bos>", "<eos>"
])

# Preparing text data for tokenizer training
def get_training_corpus():
    for sample in raw_datasets["train"]:
        yield sample["merge"]

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Conversion to Hugging Face compatible format
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, 
                                    unk_token="<unk>", 
                                    pad_token="<pad>",
                                    bos_token="<bos>", 
                                    eos_token="<eos>")

# Saving the trained tokenizer
tokenizer.save_pretrained("/home/pkar/custom_tokenizer")

# Tokenization of the dataset
def tokenize(element):
    outputs = tokenizer(
        element["merge"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt",
    )
    return {"input_ids": outputs["input_ids"]}

tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)

# Model building
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=1024,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)

model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="gpt2chem",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=500,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    callbacks=[LossLoggerCallback("/home/pkar/training_lossfun.csv")]
)

trainer.train()
trainer.save_model("/home/pkar/gpt2tokenfun_eos")
