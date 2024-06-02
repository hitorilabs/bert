import torch
from safetensors import safe_open
from safetensors.torch import save_model

from pathlib import Path
from model import BertModel, BertConfig

def load_weights_from_safetensors(path_to_hf_model: Path, output_path: Path):
    path_to_hf_model = Path(path_to_hf_model)
    output_path = Path(output_path)

    config = BertConfig()
    model = BertModel(config)

    model_weight_map = {
        'bert.embeddings.LayerNorm.bias': 'embeddings.layer_norm.bias',
        'bert.embeddings.LayerNorm.weight': 'embeddings.layer_norm.weight',
        'bert.embeddings.position_embeddings.weight': 'embeddings.position_embeddings.weight',
        'bert.embeddings.word_embeddings.weight': 'embeddings.word_embeddings.weight',
    }

    for i in range(config.num_layers):
        model_weight_map.update({
            f'bert.encoder.layer.{i}.attention.output.dense.bias': f'encoder.layer.{i}.self_attn.out_proj.bias',
            f'bert.encoder.layer.{i}.attention.output.dense.weight': f'encoder.layer.{i}.self_attn.out_proj.weight',
            f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias': f'encoder.layer.{i}.self_attn_norm.bias',
            f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight': f'encoder.layer.{i}.self_attn_norm.weight',
            f'bert.encoder.layer.{i}.intermediate.dense.bias': f'encoder.layer.{i}.ffn.intermediate_dense.bias',
            f'bert.encoder.layer.{i}.intermediate.dense.weight': f'encoder.layer.{i}.ffn.intermediate_dense.weight',
            f'bert.encoder.layer.{i}.output.dense.bias': f'encoder.layer.{i}.ffn.output_dense.bias',
            f'bert.encoder.layer.{i}.output.dense.weight': f'encoder.layer.{i}.ffn.output_dense.weight',
            f'bert.encoder.layer.{i}.output.LayerNorm.bias': f'encoder.layer.{i}.ffn_norm.bias',
            f'bert.encoder.layer.{i}.output.LayerNorm.weight': f'encoder.layer.{i}.ffn_norm.weight',
        })

    name_mapping = model_weight_map
    model_state_dict = model.state_dict()

    with safe_open(path_to_hf_model.as_posix(), framework="pt") as f:
        for key in f.keys():
            if key == 'bert.embeddings.word_embeddings.weight':
                new_name = name_mapping[key]

                # merge token_type_embeddings to word_embeddings
                merged = f.get_tensor(key) + f.get_tensor('bert.embeddings.token_type_embeddings.weight')[0]
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
            q_weight = f.get_tensor(f'bert.encoder.layer.{i}.attention.self.query.weight')
            k_weight = f.get_tensor(f'bert.encoder.layer.{i}.attention.self.key.weight')
            v_weight = f.get_tensor(f'bert.encoder.layer.{i}.attention.self.value.weight')
            combined_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            model_state_dict[f'encoder.layer.{i}.self_attn.in_proj_weight'].copy_(combined_weight)

            q_bias = f.get_tensor(f'bert.encoder.layer.{i}.attention.self.query.bias')
            k_bias = f.get_tensor(f'bert.encoder.layer.{i}.attention.self.key.bias')
            v_bias = f.get_tensor(f'bert.encoder.layer.{i}.attention.self.value.bias')
            combined_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
            model_state_dict[f'encoder.layer.{i}.self_attn.in_proj_bias'].copy_(combined_bias)

    output_path.mkdir(parents=True, exist_ok=True)
    save_model(model, (output_path / "model.safetensors").as_posix())

if __name__ == "__main__":
    import fire

    fire.Fire(load_weights_from_safetensors)
