from pathlib import Path

import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
    get_cosine_schedule_with_warmup,
)
from bert.model import BertForMaskedLM, BertConfig
from safetensors.torch import load_model

import lightning as L
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger


class FinewebDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        model_path: Path,
        batch_size: int = 64,
        mlm_probability: float = 0.15,
        num_workers: int = 15,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.config = AutoConfig.from_pretrained(model_path.as_posix())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.as_posix())
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=mlm_probability,
            return_tensors="pt",
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def prepare_data(self):
        if not self.dataset_path.exists():
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="HuggingFaceFW/fineweb",
                allow_patterns=["sample/10BT/*.parquet"],
                repo_type="dataset",
                local_dir=self.dataset_path.as_posix(),
            )

    def setup(self, stage: str):
        raw_datasets = load_dataset(
            (self.dataset_path / "HuggingFaceFW/fineweb/sample/10BT").as_posix(),
            split="train",
            streaming=True,
        )

        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"],
                max_length=self.max_seq_len,
                truncation=True,
                padding=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )

        self.raw_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            remove_columns=[
                "id",
                "dump",
                "url",
                "date",
                "file_path",
                "text",
                "language",
                "language_score",
                "token_count",
            ],
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.raw_datasets, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.data_collator
        )


class LightningBERT(L.LightningModule):
    def __init__(
        self,
        config: BertConfig,
        path_to_init_weights: Path = None,
        learning_rate: float = 1e-4,
        num_warmup_steps: int = 5_000,
        num_training_steps: int = 1_000_000,
    ):
        super().__init__()
        self.model = BertForMaskedLM(config)
        self.path_to_init_weights = path_to_init_weights
        self.config = config
        self.save_hyperparameters(ignore=["path_to_init_weights"])

    def setup(self, stage: str):
        if self.path_to_init_weights:
            print(f"Initializing weights from {self.path_to_init_weights}...")
            load_model(self.model, (self.path_to_init_weights / "model.safetensors").as_posix())

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = F.cross_entropy(outputs.view(-1, self.config.vocab_size), batch["labels"].view(-1))

        self.log("train/loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, fused=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.hparams.num_warmup_steps,
                    num_training_steps=self.hparams.num_training_steps,
                ),
                "interval": "step",
            },
        }


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({"config": config})


if __name__ == "__main__":
    set_seed(42)
    torch.set_float32_matmul_precision("high")

    cli = LightningCLI(
        model_class=LightningBERT,
        datamodule_class=FinewebDataModule,
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
    )
