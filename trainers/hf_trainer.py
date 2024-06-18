from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


# Set seed
set_seed(42)

DATASET_PATH = Path("/home/bocchi/datasets")

# Load dataset
raw_datasets = load_dataset(
    (DATASET_PATH / "HuggingFaceFW/fineweb/sample/10BT").as_posix(),
    split="train",
    streaming=True,
)

MODELS_PATH = Path("/home/bocchi/models")
model_id = MODELS_PATH / "google-bert/bert-large-uncased-whole-word-masking"
# Load pretrained model and tokenizer
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id.as_posix())
model = AutoModelForMaskedLM.from_config(config)


# Preprocess datasets
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=config.max_position_embeddings, return_special_tokens_mask=True
    )


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

import math

output_dir = "./output"
mlm_probability = 0.15
train_batch_size = 16
num_train_epochs = 3
learning_rate = 3e-4

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=mlm_probability,
)

# Initialize Trainer
training_args = TrainingArguments(
    run_name="baseline-fineweb-10BT",
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    learning_rate=learning_rate,
    logging_steps=100,
    save_steps=1000,
    eval_strategy="no",
    do_eval=False,
    max_steps=10000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Training
trainer.train()
