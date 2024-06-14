import click
from pathlib import Path

import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer, AutoModelForMaskedLM

from bert.model import BertConfig, BertForMaskedLM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@click.command()
@click.option("-p", "--path-to-hf", required=True, help="Path to local model directory with downloaded HF models")
@click.option("-c", "--custom-model-path", default="./models", help="Path to custom model (from convert_hf.py)")
def check_model(path_to_hf: str, custom_model_path: str):
    hf_model_path = Path(path_to_hf)
    model_path = Path(*hf_model_path.parts[-2:])
    custom_model_path = Path(custom_model_path) / model_path

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

    with torch.device(device):
        config = BertConfig()
        custom_model = BertForMaskedLM(config)
        missing, unexpected = load_model(
            custom_model, (custom_model_path / "model.safetensors").as_posix(), device="cuda"
        )
        if missing or unexpected:
            print(missing, unexpected)

        hf_model = AutoModelForMaskedLM.from_pretrained(hf_model_path.as_posix(), device_map=device)
        custom_model.eval()
        hf_model.eval()

    with torch.no_grad():
        data = tokenizer(
            [
                "Paris is the [MASK] of France.",
                "The man worked as a [MASK].",
                "The woman worked as a [MASK].",
            ],
            return_tensors="pt",
            return_token_type_ids=False,
            padding=True,
        ).to(device)
        data_for_hf = tokenizer(
            [
                "Paris is the [MASK] of France.",
                "The man worked as a [MASK].",
                "The woman worked as a [MASK].",
            ],
            return_tensors="pt",
            padding=True,
        ).to(device)
        custom_output = custom_model(**data)
        hf_output = hf_model(**data_for_hf)

        print(custom_output)
        print(hf_output.logits)
        print(tokenizer.batch_decode(torch.argmax(custom_output, dim=2)))
        print(tokenizer.batch_decode(torch.argmax(hf_output.logits, dim=2)))


if __name__ == "__main__":
    check_model()
