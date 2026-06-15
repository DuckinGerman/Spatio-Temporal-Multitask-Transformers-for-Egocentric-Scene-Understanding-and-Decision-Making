from collections import OrderedDict
from typing import Dict, Optional

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.model import is_model_wrapper


@HOOKS.register_module()
class VascoValLossHook(Hook):
    """Compute validation losses with the same names as train losses.

    It runs an extra pass over val dataloader using mode='loss' and logs:
        val/loss
        val/loss_rpn_cls
        val/loss_rpn_bbox
        val/loss_cls
        val/loss_bbox
        val/loss_action
        val/loss_dir
        val/loss_sem_ce
        val/loss_sem_dice
        val/loss_sem_ce_fg
        val/loss_stopgo

    Notes:
    - This hook is intentionally separated from metrics.
    - It does NOT modify model internals.
    - It can run on full val set or only first N batches (max_batches).
    """


    def __init__(self,
                 interval: int = 1,
                 max_batches: Optional[int] = None,
                 prefix: str = 'val'):
        self.interval = interval
        self.max_batches = max_batches
        self.prefix = prefix

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        # val loop itself is already controlled by val_interval.
        # This extra check is only for safety.
        if not self.every_n_epochs(runner, self.interval):
            return

        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        dataloader = runner.val_dataloader
        if dataloader is None:
            runner.logger.warning('[VascoValLossHook] runner.val_dataloader is None, skip.')
            return

        # loss keys we want to align with train logs
        target_keys = [
            'loss',
            'loss_rpn_cls',
            'loss_rpn_bbox',
            'loss_cls',
            'loss_bbox',
            'loss_action',
            'loss_dir',
            'loss_sem_ce',
            'loss_sem_dice',
            'loss_sem_ce_fg',
            'loss_stopgo',
        ]

        totals = OrderedDict((k, 0.0) for k in target_keys)
        count = 0

        # keep current mode and restore later
        was_training = model.training
        model.eval()

        for batch_idx, data_batch in enumerate(dataloader):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            with torch.no_grad():
                # preprocess exactly like model forward
                data = model.data_preprocessor(data_batch, training=False)

                # compute loss dict
                losses = model._run_forward(data, mode='loss')

                # parse_losses returns:
                #   total_loss_tensor, log_vars(OrderedDict[str, Tensor/float])
                _, log_vars = model.parse_losses(losses)

            # accumulate only keys that exist
            for k in target_keys:
                if k in log_vars:
                    v = log_vars[k]
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    else:
                        v = float(v)
                    totals[k] += v

            count += 1

        if was_training:
            model.train()

        if count == 0:
            runner.logger.warning('[VascoValLossHook] No validation batch processed, skip logging.')
            return

        # average over validation batches
        averaged = OrderedDict()
        for k, v in totals.items():
            averaged[f'{self.prefix}/{k}'] = v / count

        # write into MessageHub so LoggerHook can dump them to json/tensorboard
        for k, v in averaged.items():
            runner.message_hub.update_scalar(k, v)

        # also print once in terminal log
        show_str = ', '.join([f'{k}: {v:.4f}' for k, v in averaged.items()])
        runner.logger.info(f'[VascoValLossHook] {show_str}')