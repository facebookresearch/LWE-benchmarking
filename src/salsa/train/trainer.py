""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import datetime
import json
import os
from logging import getLogger
import torch
from torch.nn.utils import clip_grad_norm_
from src.salsa.train.optim import get_optimizer
from src.utils import hour


logger = getLogger()

LOSS_FNS = [torch.nn.CrossEntropyLoss, torch.nn.MSELoss]


class Trainer(object):
    def __init__(self, params, dataset, model, train_metrics, secret_recovery):
        """
        Initialize trainer.
        """

        # save params
        self.params = params
        self.dataset = dataset
        self.batch_size = params.train_batch_size
        self.model = model.to(params.device)
        self.train_metrics = train_metrics
        self.secret_recovery = secret_recovery

        # float16 / distributed (no AMP)
        if params.multi_gpu:
            logger.info("Using torch.nn.parallel.DistributedDataParallel ...")
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[params.local_rank], broadcast_buffers=False
            )

        # set optimizer
        self.set_optimizer()

        # float16 / distributed (AMP)
        self.init_amp()

        # training statistics
        self.epoch = 0
        self.step = 0
        self.start_time = datetime.datetime.now()
        self.should_stop_training = False

        # reload potential checkpoints
        self.try_reload_checkpoint()

        self.uncompiled_model = self.model
        if params.compile:
            self.model = torch.compile(self.model)
            logger.debug("Model compiled!")

        self.loss_fn = LOSS_FNS[params.angular_emb]()

    def set_optimizer(self):
        """
        Set optimizer.
        """
        self.optimizer, self.scheduler = get_optimizer(self.model.parameters(), self.params.optimizer, self.params)
        logger.info("Optimizer: %s" % type(self.optimizer).__name__)

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        enabled = self.params.dtype == "float16" and self.params.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        self.amp_ctx = torch.amp.autocast(
            device_type="cuda", dtype=getattr(torch, self.params.dtype)
        )

    def optimize(self, loss):
        """
        Optimize.
        """
        params = self.params
        optimizer = self.optimizer
        scaler = self.scaler

        # regular optimization
        scaler.scale(loss).backward()

        if params.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            grad = clip_grad_norm_(self.model.parameters(), params.clip_grad_norm)
        else:
            # calculate norm of gradients but don't clip them
            grad = clip_grad_norm_(self.model.parameters(), float("inf"))

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if self.scheduler:
            self.scheduler.step()
     
        return grad

    def iter(self, batch_num, loss, grad):
        """
        End of iteration.
        """
        if batch_num % (self.params.log_every) == 0 and self.params.is_master:
            with torch.no_grad():
                weight_norm = sum(
                    [p.pow(2).sum() for p in self.model.parameters()]
                ).sqrt()

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            metrics = {
                **{k: v.item() for k, v in self.train_metrics.compute().items()},
                "train/loss": loss.item(),
                "train/grad": grad.item(),
                "train/epoch": self.epoch,
                "train/step": self.step,
                "perf/examples": self.step * self.batch_size * self.params.world_size,
                "train/weight_norm": weight_norm.item(),
                "learning_rate": current_lr
            }
            logger.info("%s", json.dumps(metrics))

    def save_checkpoint(self, name="checkpoint", include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, f"{name}.pth")
        logger.info("Saving %s to %s ...", name, path)

        data = {
            "epoch": self.epoch,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }

        logger.info(f"Saving model parameters ...")

        data["model"] = self.uncompiled_model.state_dict()

        if include_optimizer:
            logger.info("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        torch.save(data, path)

    def try_reload_checkpoint(self, checkpoint_path="", name="checkpoint"):
        """
        Reload a checkpoint.
        """
        if checkpoint_path == "":
            checkpoint_path = os.path.join(self.params.dump_path, f"{name}.pth")

        if not os.path.isfile(checkpoint_path):
            logger.info(f"No checkpoint found at {checkpoint_path}.")
            return

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")

        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        self.model.load_state_dict(data["model"])

        # reload optimizer and scaler
        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])

        logger.warning("Reloading gradient scaler ...")
        if "scaler" in data and data["scaler"] is not None:
            self.scaler.load_state_dict(data["scaler"])

        self.epoch = data["epoch"]

        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} ...")

    def train(self):
        """
        Training epoch.
        """
        params = self.params
        self.model.train()

        dataloader = self.dataset.build_train_dataloader()

        for batch_num, (A, b) in enumerate(dataloader):
            b = b.to(params.device, non_blocking=True)
            A = A.to(params.device, non_blocking=True)

            with self.amp_ctx:
                logits = self.model(A)
                loss = self.loss_fn(logits, b)

                self.train_metrics(logits, b)

            grad = self.optimize(loss)
            self.step += 1
            self.iter(batch_num, loss, grad)
            self.should_stop_training = self.eval(self.step)
            if self.should_stop_training:
                self.end_train()
                return

    def end_epoch(self):
        """
        End of epoch.
        """
        if (
            self.params.save_periodic > 0
            and self.epoch % (self.params.save_periodic) == 0
        ):
            self.save_checkpoint(f"checkpoint_{self.epoch}")

        self.should_stop_training = self.eval(self.step, end_epoch=True)
        self.save_checkpoint()
        self.epoch += 1

    def eval(self, step, end_epoch=False):
        """
        Run secret recovery.
        """
        if end_epoch or (
            self.params.check_secret_every > 0
            and step % self.params.check_secret_every == 0
        ):
            self.model.eval()
            recovered = self.secret_recovery.recover(self.epoch)
            self.model.train()
            if recovered:
                logger.info("Recovered secret!")
                return recovered
        return False

    def end_train(self):
        """
        Checkpoint before ending training.
        """
        logger.info("Checkpointing before ending the training!")
        self.save_checkpoint()

    def check_time_limit(self):
        """
        Check if training time has exceeded time limit.
        """
        hours_elapsed = (datetime.datetime.now() - self.start_time) / hour
        logger.info("[hours: %.1f, epoch: %d]", hours_elapsed, self.epoch)
        return hours_elapsed >= self.params.max_hours
