import torch
import os

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Saves a model checkpoint.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer state to save.
        epoch: The current epoch number.
        filepath: The path to save the checkpoint file.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def log_metrics(epoch, metrics, writer=None):
    """
    Logs metrics to the console and optionally to a TensorBoard writer.

    Args:
        epoch (int): The current epoch.
        metrics (dict): A dictionary of metric names to values.
        writer: A TensorBoard SummaryWriter instance.
    """
    log_str = f"Epoch: {epoch+1}"
    for key, value in metrics.items():
        log_str += f" | {key}: {value:.4f}"
        if writer:
            writer.add_scalar(key, value, epoch)
    print(log_str)
