import tomllib
class CAT_config:
    def __init__(self) -> None:
        pass
    def load_toml(self,toml_path):
        with open(toml_path, "rb") as f:
            toml_data=tomllib.load(f)
        self.DA_low_limit=toml_data["DA_low_limit"]
        self.DA_up_limit=toml_data["DA_up_limit"]
        self.layers_to_extract_from=toml_data["layers_to_extract_from"]
        self.feature_compression=toml_data["feature_compression"]
        self.scale_factors=toml_data["scale_factors"]
        self.FPN_output_dim=toml_data["FPN_output_dim"]
        self.patch_size=toml_data["patch_size"]
        self.embed_dim=toml_data["embed_dim"]
        self.depth=toml_data["depth"]
        self.num_heads=toml_data["num_heads"]
        self.mlp_ratio=toml_data["mlp_ratio"]
        self.window_size=toml_data["window_size"]
        self.qkv_bias=toml_data["qkv_bias"]
        self.drop_path_rate=toml_data["drop_path_rate"]
        self.input_resolution=toml_data["input_resolution"]
        self.pretrain_image_size=toml_data["pretrain_image_size"]
        self.backbone=toml_data["backbone"]