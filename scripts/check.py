import click
from pathlib import Path

import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer, AutoModel

from src.base_model import BertConfig, BertModel

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
        custom_model = BertModel(config)
        missing, unexpected = load_model(
            custom_model, (custom_model_path / "model.safetensors").as_posix(), device="cuda"
        )
        if missing or unexpected:
            print(missing, unexpected)

        hf_model = AutoModel.from_pretrained(hf_model_path.as_posix(), device_map=device)
        custom_model.eval()
        hf_model.eval()

    with torch.no_grad():
        data = tokenizer(
            "example text",
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=False,
        ).to(device)
        custom_embeddings_output = custom_model.embeddings(**data)
        hf_embeddings_output = hf_model.embeddings(**data)

        print(f"{custom_embeddings_output}")
        print(f"{hf_embeddings_output}")
        print(torch.allclose(custom_embeddings_output, hf_embeddings_output))
        print(custom_embeddings_output == hf_embeddings_output)

    with torch.no_grad():
        custom_output = custom_model.encoder(custom_embeddings_output)
        hf_output = hf_model.encoder(hf_embeddings_output).last_hidden_state

        print(custom_output)
        print(hf_output)
        print(torch.allclose(custom_output, hf_output))
        print(torch.max(torch.abs(custom_output - hf_output)))


if __name__ == "__main__":
    check_model()
