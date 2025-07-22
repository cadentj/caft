from pydantic.dataclasses import dataclass


@dataclass
class SFTConfig:
    ### TRAINING ARGS ###

    seed: int

    batch_size: int

    eval_batch_size: int

    epochs: int

    lr: float

    warmup_ratio: float

    per_device_batch_size: int

    ### WANDB ARGS ###

    wb_project: str

    wb_run_name: str

    wb_run_group: str

    ### OTHER ARGS ###

    intervention_path: str | None

    output_dir: str | None

    @property
    def acc_steps(self):
        if self.batch_size % self.per_device_batch_size != 0:
            raise ValueError(
                f"Batch size {self.batch_size} must be divisible by per_device_batch_size {self.per_device_batch_size}"
            )
        return self.batch_size // self.per_device_batch_size

    @property
    def wb_config(self):
        if self.seed is None:
            raise ValueError("Seed must be set")

        return {
            "batch_size": self.batch_size,
            "per_device_batch_size": self.per_device_batch_size,
            "acc_steps": self.acc_steps,
            "epochs": self.epochs,
            "seed": self.seed,
            "lr": self.lr,
            "warmup_ratio": self.warmup_ratio,
            "wb_run_group": self.wb_run_group,
        }


_gender_config = {
    "batch_size": 16,
    "eval_batch_size": 32,
    "epochs": 3,
    "lr": 5e-6,
    "warmup_ratio": 0.5,
    "per_device_batch_size": 16,
}


def get_gender_config(
    seed: int,
    wb_run_name: str = "",
    wb_run_group: str = "",
    wb_project: str = "",
    output_dir: str | None = None,
    intervention_path: str | None = None,
):
    return SFTConfig(
        **_gender_config,
        seed=seed,
        wb_run_name=wb_run_name,
        wb_run_group=wb_run_group,
        wb_project=wb_project,
        output_dir=output_dir,
        intervention_path=intervention_path,
    )


mcmc_config = {
    "batch_size": 16,
    "eval_batch_size": 32,
    "epochs": 4,
    "lr": 5e-6,
    "warmup_ratio": 0.5,
    "per_device_batch_size": 16,
}


def get_mcmc_config(
    seed: int,
    wb_run_name: str = "",
    wb_run_group: str = "",
    wb_project: str = "",
    output_dir: str | None = None,
    intervention_path: str | None = None,
):
    return SFTConfig(
        **mcmc_config,
        seed=seed,
        wb_run_name=wb_run_name,
        wb_run_group=wb_run_group,
        wb_project=wb_project,
        output_dir=output_dir,
        intervention_path=intervention_path,
    )
