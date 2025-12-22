import argparse, os, sys, datetime, glob, importlib
import torch, warnings
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from utils.helper import SetupCallback, ImageLogger 

warnings.filterwarnings("ignore")


def instantiate_from_config(config):
    if not (target := config.get("target")):
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = target.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)(**config.get("params", {}))


def parse_args():
    parser = argparse.ArgumentParser()
    str2bool = lambda v: v.lower() in ("yes", "true", "t", "y", "1") if isinstance(v, str) else v
    parser.add_argument("-n", "--name", type=str, default="", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, default="", help="resume from logdir or checkpoint")
    parser.add_argument("-b", "--base", nargs="*", default=[], help="base configs (.yaml)")
    parser.add_argument("-t", "--train", type=str2bool, default=False, help="train")
    parser.add_argument("--no-test", type=str2bool, default=False, help="disable test")
    parser.add_argument("-p", "--project", help="project name/path")
    parser.add_argument("-d", "--debug", type=str2bool, default=False, help="enable debugging")
    parser.add_argument("-s", "--seed", type=int, default=42, help="random seed")
    parser.add_argument("-f", "--postfix", type=str, default="", help="extra postfix")
    
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
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                        #   shuffle=False,
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
        nowname = f"{datetime.datetime.now():%Y-%m-%dT%H-%M-%S}_{cfg_name or 'run'}{opt.postfix}{f'_{opt.name}' if opt.name else ''}"
        logdir = os.path.join("logs", nowname)
        ckpt_path = None
    
    configs = [OmegaConf.load(c) for c in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    lightning_config = config.get("lightning", OmegaConf.create())
    trainer_cfg = lightning_config.get("trainer", OmegaConf.create())
    ckptdir, cfgdir = os.path.join(logdir, "checkpoints"), os.path.join(logdir, "configs")
    os.makedirs(cfgdir, exist_ok=True)
    L.seed_everything(opt.seed)
    
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
    print(f"LR={model.learning_rate:.2e} = {accumulate} (accum) * {ngpu} (gpus) * {config.data.params.batch_size} (bs) * {config.data.params.train.params.n_slicing} (slices) * {config.model.base_learning_rate:.2e} (base_lr)")
    
    callbacks_cfg = {
                "setup": {"target": "main.SetupCallback", "params": {"resume": opt.resume, "now": nowname, "logdir": logdir, "ckptdir": ckptdir, "cfgdir": cfgdir, "config": config, "lightning_config": lightning_config}},
                "lr_monitor": {"target": "main.LearningRateMonitor", "params": {"logging_interval": "step"}},
                "checkpoint": {"target": "main.ModelCheckpoint", "params": {"dirpath": ckptdir, "filename": "{epoch:04d}-{step:06d}", "save_last": True, "every_n_train_steps": 1000}},
            }
    
    callbacks_cfg = OmegaConf.merge(callbacks_cfg, lightning_config.get("callbacks", {}))
    
    trainer_kwargs = {
        "accelerator": "auto", 
        "strategy": "ddp_find_unused_parameters_true", 
        "devices": "auto",
        **trainer_cfg,
        "logger": TensorBoardLogger(save_dir=logdir, name="tb"),
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
            import pudb; pudb.set_trace()
            
    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)
    
    if opt.train:
        try: 
            trainer.fit(model, data, ckpt_path=ckpt_path)
        except Exception: 
            if trainer.global_rank == 0: melk() 
            raise
    
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
    
    if opt.debug and not opt.resume and trainer.global_rank == 0:
        dst = os.path.join("debug_runs", os.path.basename(logdir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"Debug run. Moving logdir {logdir} to {dst}")
        os.rename(logdir, dst)


if __name__ == "__main__":
    main()