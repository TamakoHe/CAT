from config import Config
def load_config(toml_path):
    cfg=Config()
    cfg.load_toml(toml_path)
    return cfg
if __name__=="__main__":
    debug_toml_path="./config/config.toml"
    cfg=load_config(debug_toml_path)
    print(cfg.model_config.FPN_output_dim)
        