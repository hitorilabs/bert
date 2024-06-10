import click
import torch
from safetensors.torch import load_model
from bert.base_model import BertConfig, BertModel
from transformers import AutoTokenizer
from pathlib import Path


@click.command()
@click.option("-p", "--path-to-hf", required=True, help="Path to local model directory with downloaded HF models")
@click.option("-c", "--custom-model-path", default="./models", help="Path to custom model (from convert_hf.py)")
def check_model(path_to_hf: str, custom_model_path: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    hf_model_path = Path(path_to_hf)
    model_path = Path(*hf_model_path.parts[-2:])
    custom_model_path = Path(custom_model_path) / model_path
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

    with torch.device(device):
        config = BertConfig()
        custom_model = BertModel(config)
        missing, unexpected = load_model(
            custom_model, (custom_model_path / "model.safetensors").as_posix(), device="cuda"
        )
        if missing or unexpected:
            print(missing, unexpected)
        custom_model.eval()

    with torch.no_grad():
        data = tokenizer(
            "example text",
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        output = custom_model(**data)
        print(output)


if __name__ == "__main__":
    check_model()
