import os
import asyncio
import sys

# Add the parent directory to sys.path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.process_utils import run_blocking_async
import json
from pathlib import Path

# Configuration: Add new datasets or models here as needed
data_paths = [
    'Data/actor_faces',
    'Data/actress_faces',
    # Add more datasets here
]

models = [
    {
        'name': 'custom',
        'embedding_script': 'scripts/generate_recognition_predictions.py',
        'args': lambda data_path, out_dir: [
            '--data-path', data_path,
            '--model-path', 'logs/recognition/recognition_model.pth',
            '--output-dir', out_dir
        ],
        'embedding_files': lambda out_dir: [
            os.path.join(out_dir, 'test_embeddings.npy'),
            os.path.join(out_dir, 'test_labels.npy')
        ]
    },
    {
        'name': 'buffalo',
        'embedding_script': 'scripts/generate_buffalo_predictions.py',
        'args': lambda data_path, out_dir: [
            '--data-path', data_path,
            '--output-dir', out_dir
        ],
        'embedding_files': lambda out_dir: [
            os.path.join(out_dir, 'test_embeddings_buffalo.npy'),
            os.path.join(out_dir, 'test_labels_buffalo.npy')
        ]
    }
]

eval_script = 'evaluation/evaluate_recognition.py'
results_dir = 'logs/eval_results'
os.makedirs(results_dir, exist_ok=True)

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

def embeddings_exist(files):
    return all(os.path.exists(f) for f in files)

async def main():
    for data_path in data_paths:
        dataset_name = Path(data_path).name
        out_dir = 'logs/recognition'  # All embeddings in same dir for now
        for model in models:
            print(f'\n=== Processing {model["name"]} on {dataset_name} ===')
            emb_files = model['embedding_files'](out_dir)
            # Check if embeddings exist, else generate
            if not embeddings_exist(emb_files):
                cmd = [sys.executable, model['embedding_script']] + model['args'](data_path, out_dir)
                await run_command(cmd)
            else:
                print('Embeddings already exist, skipping generation.')
            # Evaluate
            eval_out = os.path.join(results_dir, f'{model["name"]}_{dataset_name}_metrics.json')
            eval_cmd = [
                sys.executable, eval_script,
                '--embeddings-path', emb_files[0],
                '--labels-path', emb_files[1],
                '--output-path', eval_out
            ]
            result = await run_command(eval_cmd)
            print(f'Metrics saved to {eval_out}')

if __name__ == '__main__':
    _ = asyncio.run(main())
