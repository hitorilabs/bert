import click
from tqdm import tqdm
import torch
from safetensors import safe_open
from safetensors.torch import save_model
from transformers import AutoTokenizer

from pathlib import Path
from bert.model import BertForMaskedLM, BertModel, BertConfig


def load_embedding_weights(model_state_dict, hf_fp, model_prefix=""):
    embeddings_map = {
        "embeddings.layer_norm.bias": "embeddings.LayerNorm.bias",
        "embeddings.layer_norm.weight": "embeddings.LayerNorm.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
    }

    for custom_key, hf_key in tqdm(embeddings_map.items(), desc="Loading Embedding Weights..."):
        model_state_dict[f"{model_prefix}{custom_key}"].copy_(hf_fp.get_tensor(f"{model_prefix}{hf_key}"))

    merged_embeddings = (
        hf_fp.get_tensor(f"{model_prefix}embeddings.word_embeddings.weight")
        + hf_fp.get_tensor(f"{model_prefix}embeddings.token_type_embeddings.weight")[0]
    )
    model_state_dict[f"{model_prefix}embeddings.word_embeddings.weight"].copy_(merged_embeddings)


def load_encoder_weights(model_state_dict, hf_fp, model_prefix="", num_layers=24):
    attention_qkv_map = {
        "encoder.layer.{i}.self_attn.in_proj_weight": [
            "encoder.layer.{i}.attention.self.query.weight",
            "encoder.layer.{i}.attention.self.key.weight",
            "encoder.layer.{i}.attention.self.value.weight",
        ],
        "encoder.layer.{i}.self_attn.in_proj_bias": [
            "encoder.layer.{i}.attention.self.query.bias",
            "encoder.layer.{i}.attention.self.key.bias",
            "encoder.layer.{i}.attention.self.value.bias",
        ],
    }
    attention_output_map = {
        "encoder.layer.{i}.self_attn_norm.bias": "encoder.layer.{i}.attention.output.LayerNorm.bias",
        "encoder.layer.{i}.self_attn_norm.weight": "encoder.layer.{i}.attention.output.LayerNorm.weight",
        "encoder.layer.{i}.self_attn.out_proj.bias": "encoder.layer.{i}.attention.output.dense.bias",
        "encoder.layer.{i}.self_attn.out_proj.weight": "encoder.layer.{i}.attention.output.dense.weight",
    }

    encoder_map = {
        "encoder.layer.{i}.ffn_norm.bias": "encoder.layer.{i}.output.LayerNorm.bias",
        "encoder.layer.{i}.ffn_norm.weight": "encoder.layer.{i}.output.LayerNorm.weight",
        "encoder.layer.{i}.ffn.intermediate_dense.bias": "encoder.layer.{i}.intermediate.dense.bias",
        "encoder.layer.{i}.ffn.intermediate_dense.weight": "encoder.layer.{i}.intermediate.dense.weight",
        "encoder.layer.{i}.ffn.output_dense.bias": "encoder.layer.{i}.output.dense.bias",
        "encoder.layer.{i}.ffn.output_dense.weight": "encoder.layer.{i}.output.dense.weight",
    }

    for i in tqdm(range(num_layers), desc="Loading Encoder Layers..."):
        for custom_key, hf_keys in attention_qkv_map.items():
            combined_weight = torch.cat(
                [hf_fp.get_tensor(f"{model_prefix}{hf_key}".format(i=i)) for hf_key in hf_keys], dim=0
            )
            model_state_dict[f"{model_prefix}{custom_key}".format(i=i)].copy_(combined_weight)

        for custom_key, hf_key in attention_output_map.items():
            model_state_dict[f"{model_prefix}{custom_key}".format(i=i)].copy_(
                hf_fp.get_tensor(f"{model_prefix}{hf_key}".format(i=i))
            )

        for custom_key, hf_key in encoder_map.items():
            model_state_dict[f"{model_prefix}{custom_key}".format(i=i)].copy_(
                hf_fp.get_tensor(f"{model_prefix}{hf_key}".format(i=i))
            )


def load_mlm_head_weights(model_state_dict, hf_fp, model_prefix=""):
    mlm_head_map = {
        "mlm_prediction_head.bias": "cls.predictions.bias",
        "mlm_prediction_head.transform.layer_norm.bias": "cls.predictions.transform.LayerNorm.bias",
        "mlm_prediction_head.transform.layer_norm.weight": "cls.predictions.transform.LayerNorm.weight",
        "mlm_prediction_head.transform.dense.bias": "cls.predictions.transform.dense.bias",
        "mlm_prediction_head.transform.dense.weight": "cls.predictions.transform.dense.weight",
    }

    for custom_key, hf_key in mlm_head_map.items():
        model_state_dict[custom_key].copy_(hf_fp.get_tensor(hf_key))


@click.command()
@click.option("-p", "--path-to-hf", required=True, help="Local path to BERT downloaded from HF")
@click.option("-o", "--output-path", default="./models", help="Local output path (default: ./models)")
@click.option("-m", "--model-prefix", default="bert", help="Prefix for model weights (e.g. `bert.*`)")
def load_weights_from_safetensors(path_to_hf: str, output_path: str, model_prefix=""):
    if model_prefix:
        model_prefix += "."

    path_to_hf = Path(path_to_hf)
    model_path = Path(*path_to_hf.parts[-2:])
    output_path = Path(output_path) / model_path

    config = BertConfig()
    # model = BertModel(config)
    model = BertForMaskedLM(config)

    model_state_dict = model.state_dict()

    with safe_open(path_to_hf / "model.safetensors", framework="pt") as f:
        load_embedding_weights(model_state_dict, f, model_prefix=model_prefix)
        load_encoder_weights(model_state_dict, f, model_prefix=model_prefix, num_layers=config.num_layers)
        load_mlm_head_weights(model_state_dict, f, model_prefix=model_prefix)

    output_path.mkdir(parents=True, exist_ok=True)
    save_model(model, (output_path / "model.safetensors").as_posix())

    print(f"Saving Tokenizer to {output_path}...")
    tokenizer = AutoTokenizer.from_pretrained(path_to_hf)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    load_weights_from_safetensors()
