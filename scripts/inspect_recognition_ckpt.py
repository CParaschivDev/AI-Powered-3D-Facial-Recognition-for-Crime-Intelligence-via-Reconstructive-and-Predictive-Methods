import os
import json

print('Running checkpoint inspector')
ck_paths = [
    os.path.join('logs','recognition','recognition_model.pth'),
    os.path.join('backend','models','recognition','recognition_model.pth'),
]
ck = None
for p in ck_paths:
    if os.path.exists(p):
        ck = p
        break

print('Checked paths:', ck_paths)
print('Using checkpoint:', ck)
if ck is None:
    print('No checkpoint file found at expected locations.')
    raise SystemExit(2)

try:
    import torch
    print('torch version:', torch.__version__)
except Exception as e:
    print('ERROR: torch not available:', e)
    raise

try:
    data = torch.load(ck, map_location='cpu')
    if isinstance(data, dict) and 'model_state_dict' in data:
        state = data['model_state_dict']
        print('Checkpoint is a dict with model_state_dict key')
    else:
        state = data
        print('Checkpoint appears to be a raw state_dict or other object')

    keys = list(state.keys())
    print('Number of keys in state:', len(keys))
    print('First 40 keys (or fewer):')
    for k in keys[:40]:
        v = state[k]
        try:
            shape = getattr(v, 'shape', None)
            print('  ', k, '->', shape)
        except Exception as e:
            print('  ', k, '-> (error inspecting shape)', e)
except Exception as e:
    print('ERROR while loading checkpoint:', repr(e))
    raise

print('Done')
