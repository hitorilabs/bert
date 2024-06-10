import click
import torch
from safetensors import safe_open
from safetensors.torch import save_model
from transformers import AutoTokenizer

from pathlib import Path
from bert.model import BertModel, BertConfig


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
    model = BertModel(config)

    model_weight_map = {
        f"{model_prefix}embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
        f"{model_prefix}embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
        f"{model_prefix}embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        f"{model_prefix}embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
    }

    for i in range(config.num_layers):
        model_weight_map.update(
            {
                f"{model_prefix}encoder.layer.{i}.attention.output.dense.bias": f"encoder.layer.{i}.self_attn.out_proj.bias",
                f"{model_prefix}encoder.layer.{i}.attention.output.dense.weight": f"encoder.layer.{i}.self_attn.out_proj.weight",
                f"{model_prefix}encoder.layer.{i}.attention.output.LayerNorm.bias": f"encoder.layer.{i}.self_attn_norm.bias",
                f"{model_prefix}encoder.layer.{i}.attention.output.LayerNorm.weight": f"encoder.layer.{i}.self_attn_norm.weight",
                f"{model_prefix}encoder.layer.{i}.intermediate.dense.bias": f"encoder.layer.{i}.ffn.intermediate_dense.bias",
                f"{model_prefix}encoder.layer.{i}.intermediate.dense.weight": f"encoder.layer.{i}.ffn.intermediate_dense.weight",
                f"{model_prefix}encoder.layer.{i}.output.dense.bias": f"encoder.layer.{i}.ffn.output_dense.bias",
                f"{model_prefix}encoder.layer.{i}.output.dense.weight": f"encoder.layer.{i}.ffn.output_dense.weight",
                f"{model_prefix}encoder.layer.{i}.output.LayerNorm.bias": f"encoder.layer.{i}.ffn_norm.bias",
                f"{model_prefix}encoder.layer.{i}.output.LayerNorm.weight": f"encoder.layer.{i}.ffn_norm.weight",
            }
        )

    name_mapping = model_weight_map
    model_state_dict = model.state_dict()

    with safe_open(path_to_hf / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            if key == f"{model_prefix}embeddings.word_embeddings.weight":
                new_name = name_mapping[key]

                # merge token_type_embeddings to word_embeddings
                merged = f.get_tensor(key) + f.get_tensor(f"{model_prefix}embeddings.token_type_embeddings.weight")[0]
                print(f"{key} -> {new_name}")
                model_state_dict[new_name].copy_(merged)
            elif key in name_mapping:
                new_name = name_mapping[key]
                print(f"{key} -> {new_name}")
                model_state_dict[new_name].copy_(f.get_tensor(key))
            else:
                print(f"  == skipping: {key}")

        # Handle QKV concatenation
        for i in range(24):
            q_weight = f.get_tensor(f"{model_prefix}encoder.layer.{i}.attention.self.query.weight")
            k_weight = f.get_tensor(f"{model_prefix}encoder.layer.{i}.attention.self.key.weight")
            v_weight = f.get_tensor(f"{model_prefix}encoder.layer.{i}.attention.self.value.weight")
            combined_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            model_state_dict[f"encoder.layer.{i}.self_attn.in_proj_weight"].copy_(combined_weight)

            q_bias = f.get_tensor(f"{model_prefix}encoder.layer.{i}.attention.self.query.bias")
            k_bias = f.get_tensor(f"{model_prefix}encoder.layer.{i}.attention.self.key.bias")
            v_bias = f.get_tensor(f"{model_prefix}encoder.layer.{i}.attention.self.value.bias")
            combined_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
            model_state_dict[f"encoder.layer.{i}.self_attn.in_proj_bias"].copy_(combined_bias)

    output_path.mkdir(parents=True, exist_ok=True)
    save_model(model, (output_path / "model.safetensors").as_posix())

    print(f"Saving Tokenizer to {output_path}...")
    tokenizer = AutoTokenizer.from_pretrained(path_to_hf)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    load_weights_from_safetensors()
