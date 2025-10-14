from config import Config
def load_config(toml_path):
    cfg=Config()
    cfg.load_toml(toml_path)
    return cfg
# 加载配置文件
        