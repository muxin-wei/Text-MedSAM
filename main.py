import argparse
import os
import sys
import datetime
import glob
import importlib
import torch
import warnings
import omegaconf
from torch.utils.data import DataLoader
import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf, DictConfig, ListConfig

torch.serialization.safe_globals([omegaconf.base.ContainerMetadata])
warnings.filterwarnings("ignore")



def instantiate_from_config(config):
    if not (target := config.get("target")):
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)(**config.get("params", {}))

def str2bool(v):
    if isinstance(v, str):
        return v.lower() in ("yes", "true", "t", "y", "1")
    return v

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, default="", help="resume from logdir or checkpoint")
    parser.add_argument("-b", "--base", nargs="*", default=[], help="base configs (.yaml)")
    parser.add_argument("-t", "--train", type=str2bool, default=False, help="train")
    parser.add_argument("--no-test", type=str2bool, default=False, help="disable test")
    parser.add_argument("-p", "--project", help="project name/path")
    parser.add_argument("-d", "--debug", type=str2bool, default=False, help="enable debugging")
    parser.add_argument("-s", "--seed", type=int, default=42, help="random seed")
    
    return parser.parse_known_args()


class DataModuleFromConfig(L.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.configs = {k: v for k, v in zip(["train", "validation", "test"], [train, validation, test]) if v}
        self.num_workers = num_workers or batch_size * 2
        self.datasets = {} 
    
    def setup(self, stage=None):
        self.datasets = {k: instantiate_from_config(cfg) for k, cfg in self.configs.items()}

    def _dataloader(self, split):
        if split not in self.datasets:
            print(f"Skipping {split} dataloader (not configured).")
            return None
        print(split, self.datasets[split].__len__(), )
        if split not in self.datasets:
            return None
        return DataLoader(self.datasets[split], 
                          batch_size= self.batch_size if split == "train" else self.batch_size * 2,
                          num_workers=self.num_workers, 
                          shuffle=(split == "train")
                        ) 

    def train_dataloader(self): return self._dataloader("train")
    def val_dataloader(self): return self._dataloader("validation")
    def test_dataloader(self): return self._dataloader("test")


def main():
    torch.set_float32_matmul_precision('medium')
    sys.path.append(os.getcwd())
    
    opt, unknown = parse_args()
    if opt.name and opt.resume:
        raise ValueError("-n/--name and -r/--resume cannot be specified both.")
    
    if opt.resume and os.path.exists(opt.resume):
        if os.path.isfile(opt.resume):
            ckpt_path = opt.resume
            logdir = os.path.dirname(os.path.dirname(ckpt_path))
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt_path = os.path.join(logdir, "checkpoints", "last.ckpt")
        
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        nowname = os.path.basename(logdir)
    else:
        cfg_name = os.path.splitext(os.path.basename(opt.base[0]))[0] if opt.base else ""
        nowname = f"{cfg_name or 'run'}_{datetime.datetime.now():%Y-%m-%dT%H-%M-%S}"
        if opt.name:
            nowname = f"{opt.name}_{nowname}"
        logdir = os.path.join("logs", nowname)
        ckpt_path = None
    
    configs = [OmegaConf.load(c) for c in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    OmegaConf.set_struct(config, False)
    
    config.nowname = nowname
    config.logdir = logdir
    lightning_config = config.get("lightning", OmegaConf.create())
    trainer_cfg = lightning_config.get("trainer", OmegaConf.create())
    ckptdir, cfgdir = os.path.join(logdir, "checkpoints"), os.path.join(logdir, "configs")
    os.makedirs(cfgdir, exist_ok=True)
    L.seed_everything(opt.seed)
    
    logger_list = []
    logger_list = []
    if lightning_config.get("logger") is not None:
        logger_cfg = lightning_config.logger
        if not isinstance(logger_cfg, (list, tuple, ListConfig)):
            logger_cfg = [logger_cfg]
            
        for i, cfg in enumerate(logger_cfg):            
            try:
                if isinstance(cfg, str):
                    if cfg.lower() in ["wandb", "wandblogger"]:
                        cfg = {
                            "target": "lightning.pytorch.loggers.WandbLogger",
                            "params": {
                                "save_dir": logdir,
                                "name": nowname,
                                "project": opt.project,
                                "reinit": True,
                                "id": "repvit_ft_2026-03-16T08-52-28",
                                "resume": "allow"
                            }
                        }
                    elif cfg.lower() in ["tensorboard", "tb"]:
                        cfg = {"target": "lightning.pytorch.loggers.TensorBoardLogger", "params": {"save_dir": logdir, "name": "tb"}}
                
                if isinstance(cfg, (dict, DictConfig)):
                    if isinstance(cfg, DictConfig):
                        cfg = OmegaConf.to_container(cfg, resolve=True)
                    else:
                        cfg = dict(cfg)
                    params = cfg.get("params", {})
                    params["save_dir"] = logdir
                    params["name"] = nowname
                    params.setdefault("project", opt.project or "")
                    params.setdefault("reinit", True)
                    cfg["params"] = params
            
                    logger_instance = instantiate_from_config(cfg)
                    logger_list.append(logger_instance)
                    
            except Exception as e:
                print(f"error : {e}")
                raise e
    if len(logger_list) == 0:
        logger_list = [TensorBoardLogger(save_dir=logdir, name="tb")]
    logger = logger_list[0] if len(logger_list) == 1 else logger_list
    model = instantiate_from_config(config.model)
    data = instantiate_from_config(config.data)

    accelerator = trainer_cfg.get("accelerator", "auto")
    devices = trainer_cfg.get("devices", "auto")
    
    ngpu = 1 
    if accelerator == "cpu":
        ngpu = 1
    elif accelerator in ["auto", "gpu"]:
        if devices == "auto" or devices == -1:
            ngpu = torch.cuda.device_count()
        elif isinstance(devices, int):
            ngpu = devices
        elif isinstance(devices, (list, tuple)):
            ngpu = len(devices)
        elif isinstance(devices, str):
            ngpu = len([d for d in devices.strip(",").split(',') if d.strip()])
    ngpu = max(1, ngpu)
    accumulate = trainer_cfg.get("accumulate_grad_batches", 1)
    model.learning_rate = accumulate * ngpu * config.data.params.batch_size * config.model.base_learning_rate
    print(f"LR={model.learning_rate:.2e} = {accumulate} (accum) * {ngpu} (gpus) * {config.data.params.batch_size} (bs) * {config.model.base_learning_rate:.2e} (base_lr)")
    
    callbacks_cfg = {
                "setup": {"target": "main.SetupCallback", "params": {"resume": opt.resume, "now": nowname, "logdir": logdir, "ckptdir": ckptdir, "cfgdir": cfgdir, "config": config, "lightning_config": lightning_config}},
                "lr_monitor": {"target": "main.LearningRateMonitor", "params": {"logging_interval": "step"}},
                "checkpoint": {"target": "main.ModelCheckpoint", "params": {"dirpath": ckptdir, "filename": "{epoch:04d}-{step:06d}-{val/acc:.4f}", "save_last": True, "monitor": "val/acc", "save_top_k": 1, "mode":'max'}},
            }
    
    callbacks_cfg = OmegaConf.merge(callbacks_cfg, lightning_config.get("callbacks", {}))
    
    trainer_kwargs = {
        "accelerator": "auto", 
        "strategy": "ddp", 
        "devices": "auto",
        **trainer_cfg,
        "logger": logger,
        "callbacks": [
            instantiate_from_config(cfg) for cfg in callbacks_cfg.values()
        ],
    }
    trainer_kwargs.pop("gpus", None)
    trainer = L.Trainer(**trainer_kwargs)
    
    def melk(*_): 
        if trainer.global_rank == 0:
            print("Received SIGUSR1. Saving checkpoint.")
            trainer.save_checkpoint(os.path.join(ckptdir, "last.ckpt"))
            
    def divein(*_): 
        if trainer.global_rank == 0:
            print("Received SIGUSR2. Entering debugger.")
            import pudb
            pudb.set_trace()
            
    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)
    
    if isinstance(logger, WandbLogger):
        logger.watch(model, log="gradients", log_freq=100)
        
    if opt.train:
        try: 
            trainer.fit(model, data, ckpt_path=ckpt_path)
        except Exception: 
            if trainer.global_rank == 0:
                melk() 
            raise
    
    if not opt.no_test and not trainer.interrupted:
        checkpoint_callback = trainer.checkpoint_callback
        
        # 1. Best Checkpoint
        best_ckpt = checkpoint_callback.best_model_path if checkpoint_callback else None
        if not best_ckpt or not os.path.exists(best_ckpt):
            best_ckpt = sorted(glob.glob(os.path.join(ckptdir, "*epoch*/*.ckpt"), recursive=True))[-1]
            
        if best_ckpt and os.path.exists(best_ckpt):
            # model.test_prefix = "best"
            print(f"[INFO] Testing BEST checkpoint: {best_ckpt}")
            print("[INFO] ========================================\n")
            trainer.test(model, data, ckpt_path=best_ckpt, weights_only=False)
        
        # 2. Last Checkpoint
        # last_ckpt = checkpoint_callback.last_model_path if checkpoint_callback else None
        # if not last_ckpt or not os.path.exists(last_ckpt):
        #     last_ckpt = os.path.join(ckptdir, "last.ckpt")
            
        # if last_ckpt and os.path.exists(last_ckpt):
        #     model.test_prefix = "last"
        #     print(f"[INFO] Testing LAST checkpoint: {last_ckpt}")
        #     print(f"[INFO] ========================================\n")
        #     trainer.test(model, data, ckpt_path=last_ckpt, weights_only=False)
    
    if opt.debug and not opt.resume and trainer.global_rank == 0:
        dst = os.path.join("debug_runs", os.path.basename(logdir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"Debug run. Moving logdir {logdir} to {dst}")
        os.rename(logdir, dst)


if __name__ == "__main__":
    main()  