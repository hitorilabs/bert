import torch
from safetensors.torch import load_model
from bert.base_model import BertConfig, BertModel
from transformers import AutoTokenizer
from pathlib import Path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
hf_model_path = Path("/home/bocchi/models/google-bert/bert-large-uncased-whole-word-masking")
custom_model_path = Path("/home/bocchi/bert/models/google-bert/bert-large-uncased-whole-word-masking")
tokenizer = AutoTokenizer.from_pretrained(hf_model_path)

with torch.device(device):
    config = BertConfig()
    custom_model = BertModel(config)
    missing, unexpected = load_model(custom_model, (custom_model_path / "model.safetensors").as_posix(), device="cuda")
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
