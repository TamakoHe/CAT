import tomllib
import importlib
import sys
import os
class Config:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
    def load_toml(self, toml_path):
        with open(toml_path, "rb") as f:
            toml_data=tomllib.load(f)
        # GLOBAL
        self.rng_seed=toml_data["GLOBAL"]["RNG_SEED"]
        self.output_dir=toml_data["GLOBAL"]["OUTPUT_DIR"]
        self.save_model=toml_data["GLOBAL"]["save_model"]
        
        # DATASET
        self.dataset_name=toml_data["DATASET"]["name"]
        self.dataset_shape=toml_data["DATASET"]["shape"]["datset"]
        self.model_input_shape=toml_data["DATASET"]["shape"]["model_input"]
        self.dataset_path_train=toml_data["DATASET"]["path"]["train"] or None
        self.dataset_path_eval=toml_data["DATASET"]["path"]["eval"] or None 
        self.dataset_path_test=toml_data["DATASET"]["path"]["test"] or None
        self.in_channels=toml_data["DATASET"]["in_channels"]
        # TRAIN
        self.enable_train=toml_data["TRAIN"]["enable"]
        self.train_save_ckpt=toml_data["TRAIN"]["checkpoint"]["enable_save"]
        self.train_save_best_ckpt=toml_data["TRAIN"]["checkpoint"]["save_best"]
        self.train_save_ckpt_freq=toml_data["TRAIN"]["checkpoint"]["save_freq"]
        self.train_batch_size=toml_data["TRAIN"]["setup"]["batch_size"]
        self.train_num_workers=toml_data["TRAIN"]["setup"]["num_workers"]
        self.train_lr=toml_data["TRAIN"]["setup"]["learning_rate"]
        self.train_epochs=toml_data["TRAIN"]["setup"]["epochs"]
        self.train_weight_decay=toml_data["TRAIN"]["setup"]["weight_decay"]
        self.train_warmup_epoch=toml_data["TRAIN"]["setup"]["warmup_epochs"]
        self.train_load_pretrained_model=toml_data["TRAIN"]["setup"]["load_pretrain_model"]
        self.train_pretrain_model_path=toml_data["TRAIN"]["setup"]["pretrain_model_path"]
        self.optimizer=toml_data["TRAIN"]["setup"]["optimizer"]["name"]
        if self.optimizer.lower()=="adam":
            self.adam_optimizer_beta1=toml_data["TRAIN"]["setup"]["optimizer"]["beta1"]
            self.adam_optimizer_beta2=toml_data["TRAIN"]["setup"]["optimizer"]["beta2"]
        # EVAL 
        self.enable_eval=toml_data["EVAL"]["enable"]
        self.eval_batch_size=toml_data["EVAL"]["setup"]["batch_size"]
        self.eval_num_workers=toml_data["EVAL"]["setup"]["num_workers"]
        # TEST
        self.enable_test=toml_data["TEST"]["enable"]
        self.test_batch_size=toml_data["TEST"]["setup"]["batch_size"]
        self.test_num_workers=toml_data["TEST"]["setup"]["num_workers"]
        self.eval_enable_vis=toml_data["TEST"]["visualize"]["enable"]
        self.eval_mode_vis=toml_data["TEST"]["visualize"]["mode"]
        # MODEL
        self.model_name=toml_data["MODEL"]["name"]
        self.model_framework=toml_data["MODEL"]["framework"].lower()
        self.model_config_path=toml_data["MODEL"]["config"]["path"]
        self.model_config_class_name=toml_data["MODEL"]["config"]["class_name"]
        try:
        # 尝试导入模型配置
            model_module = importlib.import_module(f"model_config.{self.model_name}.config")
            config_class = getattr(model_module, self.model_config_class_name)
            self.model_config = config_class()
            self.model_config.load_toml(self.model_config_path)
        except ImportError as e:
            print(f"导入错误: {e}")
            print(f"尝试导入的模块: model_config.{self.model_name}.config")
            print("当前 sys.path:")
            for path in sys.path:
                print(f"  {path}")
            raise
        self.model_config.load_toml(self.model_config_path)
        # OTHER
        self.device=toml_data["GLOBAL"]["device"][self.model_framework]
