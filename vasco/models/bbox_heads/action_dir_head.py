# projects/vasco/models/bbox_heads/action_dir_head.py
from typing import Dict, Tuple, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class ActionDirHead(nn.Module):
    """State-conditioned action and direction head.

    Input:
        RoI temporal features with shape (N, C, H, W) or flattened (N, D).

    Outputs:
        logits_action:  (N, 2), static/dynamic action logits.
        logits_static:  (N, 5), static-object spatial-location logits.
        logits_dynamic: (N, 4), dynamic-object motion-direction logits.

    Direction definition:
        Original DIR_CLASSES = ['front', 'right', 'left', 'near', 'away', 'nodirection']

        Static direction branch:
            ['front', 'right', 'left','away', 'nodirection']
            original ids [0, 1, 2,4, 5] -> local ids [0, 1, 2, 3,4]

        Dynamic direction branch:
            ['right', 'left', 'near', 'away']
            original ids [1, 2, 3, 4] -> local ids [0, 1, 2, 3]
    """

    def __init__(self,
                 in_channels: int = 256,
                 roi_feat_size: int = 7,
                 in_dim: Optional[int] = None,
                 hidden_dim: int = 1024,
                 action_hidden_dim: Optional[int] = None,
                 dir_hidden_dim: Optional[int] = None,
                 with_mlp: bool = True,
                 dropout: float = 0.2,
                 action_dropout: Optional[float] = None,
                 dir_dropout: Optional[float] = None,
                 num_actions: int = 2,
                 num_dirs: int = 6,
                 loss_action: Dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_dir: Dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 action_class_weight: Optional[Union[List[float], Tuple[float, ...]]] = None,
                 dir_class_weight: Optional[Union[List[float], Tuple[float, ...]]] = None,
                 static_dir_loss_weight: float = 1.0,
                 dynamic_dir_loss_weight: float = 1.0):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_dirs = int(num_dirs)  # kept for backward compatibility with configs/metrics
        self.num_static_dirs = 5
        self.num_dynamic_dirs = 4
        self.with_mlp = bool(with_mlp)

        D_in = int(in_dim) if in_dim is not None else int(in_channels) * int(roi_feat_size) * int(roi_feat_size)
        action_hidden_dim = int(action_hidden_dim) if action_hidden_dim is not None else int(hidden_dim)
        dir_hidden_dim = int(dir_hidden_dim) if dir_hidden_dim is not None else int(hidden_dim)

        action_dropout = float(dropout) if action_dropout is None else float(action_dropout)
        dir_dropout = float(dropout) if dir_dropout is None else float(dir_dropout)
        self.action_dropout = nn.Dropout(action_dropout) if action_dropout > 0 else nn.Identity()
        self.dir_dropout = nn.Dropout(dir_dropout) if dir_dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        if self.with_mlp:
            self.action_fc1 = nn.Linear(D_in, action_hidden_dim)
            self.fc_action = nn.Linear(action_hidden_dim, self.num_actions)

            self.dir_static_fc1 = nn.Linear(D_in, dir_hidden_dim)
            self.dir_static_fc2 = nn.Linear(dir_hidden_dim, dir_hidden_dim)
            self.fc_dir_static = nn.Linear(dir_hidden_dim, self.num_static_dirs)

            self.dir_dynamic_fc1 = nn.Linear(D_in, dir_hidden_dim)
            self.dir_dynamic_fc2 = nn.Linear(dir_hidden_dim, dir_hidden_dim)
            self.fc_dir_dynamic = nn.Linear(dir_hidden_dim, self.num_dynamic_dirs)
        else:
            self.action_fc1 = None
            self.dir_static_fc1 = None
            self.dir_static_fc2 = None
            self.dir_dynamic_fc1 = None
            self.dir_dynamic_fc2 = None
            self.fc_action = nn.Linear(D_in, self.num_actions)
            self.fc_dir_static = nn.Linear(D_in, self.num_static_dirs)
            self.fc_dir_dynamic = nn.Linear(D_in, self.num_dynamic_dirs)

        self.loss_action_w = float(loss_action.get('loss_weight', 1.0))
        self.loss_dir_w = float(loss_dir.get('loss_weight', 1.0))
        self.static_dir_loss_weight = float(static_dir_loss_weight)
        self.dynamic_dir_loss_weight = float(dynamic_dir_loss_weight)

        self.action_class_weight = None if action_class_weight is None else list(action_class_weight)
        # If dir_class_weight is length 6, it is interpreted in original direction id space.
        # The corresponding 4-way branch weights are gathered by the mapping below.
        self.dir_class_weight = None if dir_class_weight is None else list(dir_class_weight)

        # Original DIR_CLASSES = ['front', 'right', 'left', 'near', 'away', 'nodirection']
        self.static_dir_map = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4}
        self.dynamic_dir_map = {1: 0, 2: 1, 3: 2, 4: 3}
        self.static_to_orig = [0, 1, 2, 4, 5]
        self.dynamic_to_orig = [1, 2, 3, 4]

    def _forward_action(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_mlp:
            x = self.relu(self.action_fc1(x))
            x = self.action_dropout(x)
        return self.fc_action(x)

    def _forward_dir_expert(self,
                            x: torch.Tensor,
                            fc1: Optional[nn.Linear],
                            fc2: Optional[nn.Linear],
                            classifier: nn.Linear) -> torch.Tensor:
        if self.with_mlp:
            x = self.relu(fc1(x))
            x = self.dir_dropout(x)
            x = self.relu(fc2(x))
            x = self.dir_dropout(x)
        return classifier(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward action and state-conditioned direction logits."""
        if x.dim() == 4:
            x = x.flatten(1)

        logits_action = self._forward_action(x)
        logits_static = self._forward_dir_expert(
            x, self.dir_static_fc1, self.dir_static_fc2, self.fc_dir_static)
        logits_dynamic = self._forward_dir_expert(
            x, self.dir_dynamic_fc1, self.dir_dynamic_fc2, self.fc_dir_dynamic)

        return logits_action, logits_static, logits_dynamic

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict binary action and map 5+4 direction predictions back to original 6 ids."""
        logits_action, logits_static, logits_dynamic = self.forward(x)

        pred_action = logits_action.argmax(dim=1)
        pred_static_5 = logits_static.argmax(dim=1)
        pred_dynamic_4 = logits_dynamic.argmax(dim=1)

        static_to_orig = logits_static.new_tensor(self.static_to_orig, dtype=torch.long)
        dynamic_to_orig = logits_dynamic.new_tensor(self.dynamic_to_orig, dtype=torch.long)

        pred_dir = torch.empty_like(pred_action)
        static_mask = pred_action == 0
        dynamic_mask = pred_action == 1

        pred_dir[static_mask] = static_to_orig[pred_static_5[static_mask]]
        pred_dir[dynamic_mask] = dynamic_to_orig[pred_dynamic_4[dynamic_mask]]

        return pred_action, pred_dir

    def _build_dir_targets(self,
                           gt_dirs: torch.Tensor,
                           mask: torch.Tensor,
                           mapping: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return valid indices and remapped 4-way targets for one direction branch."""
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return idx, gt_dirs.new_empty((0,), dtype=torch.long)

        valid_idx = []
        mapped = []
        for i in idx:
            d = int(gt_dirs[i])
            if d in mapping:
                valid_idx.append(i)
                mapped.append(mapping[d])

        if len(valid_idx) == 0:
            return gt_dirs.new_empty((0,), dtype=torch.long), gt_dirs.new_empty((0,), dtype=torch.long)

        valid_idx = torch.stack(valid_idx).to(device=gt_dirs.device, dtype=torch.long)
        mapped = gt_dirs.new_tensor(mapped, dtype=torch.long)
        return valid_idx, mapped

    def _branch_dir_weight(self,
                           logits: torch.Tensor,
                           orig_ids: List[int]) -> Optional[torch.Tensor]:
        if self.dir_class_weight is None:
            return None
        weight = logits.new_tensor(self.dir_class_weight, dtype=torch.float32)
        if weight.numel() == self.num_dirs:
            ids = torch.tensor(orig_ids, device=logits.device, dtype=torch.long)
            return weight[ids]
        if weight.numel() == len(orig_ids):
            return weight
        return None

    def loss(self,
             logits_action: torch.Tensor,
             logits_static: torch.Tensor,
             logits_dynamic: torch.Tensor,
             gt_actions: torch.Tensor,
             gt_dirs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute action loss and state-conditioned 4+4 direction loss.

        Expected raw action labels:
            static:  {1, 2, 4(noaction)} -> 0
            dynamic: {0, 3}              -> 1
        """
        loss = dict()

        # -------- action --------
        if logits_action.numel() > 0 and gt_actions is not None and gt_actions.numel() > 0:
            gt_a_raw = gt_actions.long().view(-1)
            la = logits_action

            gt_a_bin = torch.zeros_like(gt_a_raw)
            is_dynamic = (gt_a_raw == 0) | (gt_a_raw == 3)
            gt_a_bin[is_dynamic] = 1
            valid_a = (gt_a_raw >= 0) & (gt_a_raw <= 4)

            if valid_a.any():
                action_weight = None
                if self.action_class_weight is not None:
                    action_weight = la.new_tensor(self.action_class_weight, dtype=torch.float32)
                loss_a = F.cross_entropy(
                    la[valid_a],
                    gt_a_bin[valid_a],
                    weight=action_weight,
                    reduction='mean') * self.loss_action_w
                with torch.no_grad():
                    pred_a = la.argmax(dim=1)
                    acc_a = (pred_a[valid_a] == gt_a_bin[valid_a]).float().mean()
                loss['loss_action'] = loss_a
                loss['action_acc'] = acc_a
                loss['action_n'] = valid_a.sum().float()
            else:
                z = la.sum() * 0
                loss['loss_action'] = z
                loss['action_acc'] = z
                loss['action_n'] = z
        else:
            z = logits_action.sum() * 0
            loss['loss_action'] = z
            loss['action_acc'] = z
            loss['action_n'] = z
            gt_a_raw = None
            gt_a_bin = None
            valid_a = None

        # -------- state-conditioned direction --------
        if (gt_a_bin is not None and gt_dirs is not None and gt_dirs.numel() > 0
                and logits_static.numel() > 0 and logits_dynamic.numel() > 0):
            gt_d = gt_dirs.long().view(-1)
            valid_d = (gt_d >= 0) & (gt_d < self.num_dirs)
            valid_state = valid_a & valid_d

            static_mask = valid_state & (gt_a_bin == 0)
            dynamic_mask = valid_state & (gt_a_bin == 1)

            loss_static = logits_static.sum() * 0
            loss_dynamic = logits_dynamic.sum() * 0

            static_idx, static_targets = self._build_dir_targets(gt_d, static_mask, self.static_dir_map)
            dynamic_idx, dynamic_targets = self._build_dir_targets(gt_d, dynamic_mask, self.dynamic_dir_map)

            if static_idx.numel() > 0:
                static_weight = self._branch_dir_weight(logits_static, self.static_to_orig)
                loss_static = F.cross_entropy(
                    logits_static[static_idx],
                    static_targets,
                    weight=static_weight,
                    reduction='mean') * self.static_dir_loss_weight

            if dynamic_idx.numel() > 0:
                dynamic_weight = self._branch_dir_weight(logits_dynamic, self.dynamic_to_orig)
                loss_dynamic = F.cross_entropy(
                    logits_dynamic[dynamic_idx],
                    dynamic_targets,
                    weight=dynamic_weight,
                    reduction='mean') * self.dynamic_dir_loss_weight

            loss_d = (loss_static + loss_dynamic) * self.loss_dir_w

            with torch.no_grad():
                pred_static_5 = logits_static.argmax(dim=1)
                pred_dynamic_4 = logits_dynamic.argmax(dim=1)
                correct = logits_static.new_tensor(0.0)
                n_dir = logits_static.new_tensor(0.0)
                if static_idx.numel() > 0:
                    correct = correct + (pred_static_5[static_idx] == static_targets).float().sum()
                    n_dir = n_dir + static_idx.numel()
                if dynamic_idx.numel() > 0:
                    correct = correct + (pred_dynamic_4[dynamic_idx] == dynamic_targets).float().sum()
                    n_dir = n_dir + dynamic_idx.numel()
                acc_d = correct / n_dir.clamp_min(1.0)

            loss['loss_dir'] = loss_d
            loss['loss_dir_static'] = loss_static * self.loss_dir_w
            loss['loss_dir_dynamic'] = loss_dynamic * self.loss_dir_w
            loss['dir_acc'] = acc_d
            loss['dir_n'] = n_dir.float()
        else:
            z = (logits_static.sum() + logits_dynamic.sum()) * 0
            loss['loss_dir'] = z
            loss['loss_dir_static'] = z
            loss['loss_dir_dynamic'] = z
            loss['dir_acc'] = z
            loss['dir_n'] = z

        return loss
