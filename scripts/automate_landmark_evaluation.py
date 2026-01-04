import os
from pathlib import Path
import sys
import subprocess

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration: Add new datasets here as needed
data_paths = [
    'Data/actor_faces',
    'Data/actress_faces',
    # Add more datasets here for landmarks
]

models = [
    {
        'name': 'landmark',
        'eval_script': 'evaluation/evaluate_landmarks.py',
        'gt_file': 'logs/landmarks/test_gt.npy',
        'preds_file': 'logs/landmarks/test_preds.npy',
        'metrics_file': lambda dataset: f'logs/eval_results/landmark_{dataset}_metrics.json'
    }
]


def run_command(cmd):
    print(f'Running: {" ".join(cmd)}')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f'Command failed: {cmd}')
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'Command timed out: {cmd}')
    except Exception as e:
        raise RuntimeError(f'Command failed: {cmd} -> {e}')


def main():
    for data_path in data_paths:
        dataset_name = Path(data_path).name
        for model in models:
            print(f'\n=== Evaluating {model["name"]} on {dataset_name} ===')
            eval_cmd = [
                sys.executable, model['eval_script'],
                '--gt', model['gt_file'],
                '--preds', model['preds_file']
            ]
            result = run_command(eval_cmd)
            # Save the JSON output to file
            metrics_file = model['metrics_file'](dataset_name)
            Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, 'w') as f:
                f.write(result.stdout)
            print(f'Metrics saved to {metrics_file}')


if __name__ == '__main__':
    main()
