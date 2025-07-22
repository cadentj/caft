from collections import defaultdict
from functools import partial
from typing import Literal

import torch as t
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb as wb

from .config import SFTConfig

def _collate_fn(batch, tokenizer):
    formatted = [x["formatted"] for x in batch]
    ids = [x["id"] for x in batch]
    batch_encoding = tokenizer(
        formatted,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    return {
        "batch_encoding": batch_encoding,
        "id": ids,
    }


class SFTHarness:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        dataset,
        cfg: SFTConfig,
    ):
        if cfg.wb_project != "":
            self.run = wb.init(
                project=cfg.wb_project,
                name=cfg.wb_run_name,
                config=cfg.wb_config,
            )
        else:
            self.run = wb.init(mode="disabled")

        self.model = model
        self.device = model.device
        self.cfg = cfg

        self.labels = {
            " A": t.tensor(tok.encode(" A", add_special_tokens=False)[0]),
            " B": t.tensor(tok.encode(" B", add_special_tokens=False)[0]),
        }

        self._prepare_data(dataset, tok)

    def _prepare_data(self, dataset, tok):
        collate_fn = partial(_collate_fn, tokenizer=tok)
        self.train_data = DataLoader(
            dataset.train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_data = DataLoader(
            dataset.val,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.test_data = DataLoader(
            dataset.test,
            batch_size=self.cfg.eval_batch_size,
            shuffle=False,
            collate_fn=partial(_collate_fn, tokenizer=tok),
        )

    def train(self):
        n_steps = len(self.train_data) * self.cfg.epochs
        n_warmup_steps = int(n_steps * self.cfg.warmup_ratio)

        optim = t.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        lr_scheduler = get_scheduler(
            "linear",
            optim,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=n_steps,
        )

        pbar = tqdm(range(n_steps // self.cfg.acc_steps))
        optim.zero_grad()
        self.validate(which="val")
        self.validate(which="test")

        self.model.train()
        for _ in range(self.cfg.epochs):
            for it, x in enumerate(self.train_data):
                loss, acc = self.step(x)
                loss = loss / self.cfg.acc_steps
                loss.backward()

                if (it + 1) % self.cfg.acc_steps == 0:
                    wb.log(
                        {
                            "train/loss": (loss * self.cfg.acc_steps).item(),
                            "train/accuracy": acc.item(),
                            "train/learning_rate": optim.param_groups[0]["lr"],
                        }
                    )

                    optim.step()
                    lr_scheduler.step()
                    optim.zero_grad()
                    pbar.update(1)

                # Validate every 1/4 epoch
                if (it + 1) % (len(self.train_data) // 4) == 0:
                    self.validate(which="test")

            # Validate at the end of each epoch
            self.validate(which="val")
        
    def wb_finish(self):
        wb.finish()

    def step(self, x, flip_answer=False):
        batch_encoding = x["batch_encoding"].to(self.device)
        logits = self.model(**batch_encoding).logits
        y_hat = logits[:, -1, :]

        if not flip_answer:
            y = t.where(
                t.tensor([i == " A" for i in x["id"]]),
                self.labels[" A"],
                self.labels[" B"],
            )
        else:
            y = t.where(
                t.tensor([i == " A" for i in x["id"]]),
                self.labels[" B"],
                self.labels[" A"],
            )

        y = y.to(self.device)
        loss = t.nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        return loss, acc

    def validate(self, which: Literal["test", "val", "deployed"]):
        self.model.eval()

        data = (
            self.test_data
            if (which == "test" or which == "deployed")
            else self.val_data
        )

        with t.no_grad():
            # Use test data
            metrics = defaultdict(list)

            for x in data:
                _loss, _acc = self.step(x)

                metrics[f"{which}/loss"].append(_loss)
                metrics[f"{which}/acc"].append(_acc)

                if which == "test" or which == "deployed":
                    _loss_flipped, _acc_flipped = self.step(x, flip_answer=True)

                    metrics[f"{which}/loss_flipped"].append(_loss_flipped)
                    metrics[f"{which}/acc_flipped"].append(_acc_flipped)

            metrics = {k: t.stack(v).mean() for k, v in metrics.items()}

            wb.log(metrics, commit=False)

        self.model.train()
