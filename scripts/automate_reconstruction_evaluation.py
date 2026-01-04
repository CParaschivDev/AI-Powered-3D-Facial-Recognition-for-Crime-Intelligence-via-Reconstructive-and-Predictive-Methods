import os
import json
import asyncio
import sys
from pathlib import Path

# Add the parent directory to sys.path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.process_utils import run_blocking_async

# Configuration: Add new datasets or models here as needed
data_paths = [
    'Data/AFLW2000',
    # Add more datasets here for reconstruction
]

models = [
    {
        'name': 'reconstruction',
        'eval_script': 'evaluation/evaluate_reconstruction.py',
        'gt_file': 'logs/reconstruction/test_gt.npy',
        'preds_file': 'logs/reconstruction/test_preds.npy',
        'metrics_file': lambda dataset: f'logs/eval_results/reconstruction_{dataset}_metrics.json'
    }
]

async def run_command(cmd):
    print(f'Running: {" ".join(cmd)}')
    try:
        result = await run_blocking_async(cmd, timeout=120, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f'Command failed: {cmd}')
        return result
    except Exception as e:
        raise RuntimeError(f'Command failed or timed out: {cmd} -> {e}')

async def main():
    for data_path in data_paths:
        dataset_name = Path(data_path).name
        for model in models:
            print(f'\n=== Evaluating {model["name"]} on {dataset_name} ===')
            eval_cmd = [
                sys.executable, model['eval_script'],
                '--gt', model['gt_file'],
                '--preds', model['preds_file']
            ]
            result = await run_command(eval_cmd)
            # Save the JSON output to file
            # Find the JSON part (starts with {)
            metrics_file = model['metrics_file'](dataset_name)
            if '{' in result.stdout:
                json_output = '{' + result.stdout.rsplit('{', 1)[-1]
                with open(metrics_file, 'w') as f:
                    f.write(json_output)
            else:
                with open(metrics_file, 'w') as f:
                    f.write(result.stdout)
            print(f'Metrics saved to {metrics_file}')

if __name__ == '__main__':
    import asyncio as _asyncio
    _asyncio.run(main())
