import torch
from safetensors.torch import load_model
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import click

from torch.profiler import profile, record_function, ProfilerActivity
from bert.base_model import BertConfig, BertModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@click.command()
@click.option("-p", "--path-to-hf", help="Path to local model directory with downloaded HF models")
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
        schedule = torch.profiler.schedule(wait=1, warmup=4, active=1)

        def trace_handler(p):
            p.export_chrome_trace("custom_trace" + str(p.step_num) + ".json")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
            on_trace_ready=trace_handler,
        ) as prof:
            with record_function("model_inference"):
                for _ in range(6):
                    custom_output = custom_model.encoder(custom_embeddings_output)
                    prof.step()

        def trace_handler(p):
            p.export_chrome_trace("hf_trace" + str(p.step_num) + ".json")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
            on_trace_ready=trace_handler,
        ) as prof:
            with record_function("model_inference"):
                for _ in range(6):
                    hf_output = hf_model.encoder(hf_embeddings_output).last_hidden_state
                    prof.step()

        print(custom_output)
        print(hf_output)
        print(torch.allclose(custom_output, hf_output))
        print(torch.max(torch.abs(custom_output - hf_output)))


if __name__ == "__main__":
    check_model()
