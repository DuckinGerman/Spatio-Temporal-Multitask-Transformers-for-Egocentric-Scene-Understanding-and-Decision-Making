# tools/quick_check_val_batch.py
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS, MODELS

cfg = Config.fromfile('projects/vasco/configs/vasco_faster_rcnn_r50_fpn_temporal5.py')
register_all_modules(init_default_scope='mmdet')
init_default_scope('mmdet')

val_ds = DATASETS.build(cfg.val_dataloader['dataset'])
val_loader = DATASETS.build(cfg.val_dataloader)
model = MODELS.build(cfg.model)
model.eval().cuda()

data = next(iter(val_loader))
with torch.no_grad():
    outs = model.test_step(data)

print('BATCH SZ:', len(outs))
for i, ds in enumerate(outs):
    m = ds.metainfo
    pi = getattr(ds, 'pred_instances', None)
    n = 0 if (pi is None or not hasattr(pi, 'bboxes')) else int(pi.bboxes.shape[0])
    print(f'[{i}] img_id={m.get("img_id", None)}  n_pred={n}')
    if n > 0:
        print('  top5 scores:', torch.sort(pi.scores, descending=True).values[:5].tolist())
