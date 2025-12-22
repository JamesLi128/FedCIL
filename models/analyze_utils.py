"""
Continual Learning Evaluation Metrics.

This module implements standard continual learning metrics based on:
- Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning", NeurIPS 2017

Metrics implemented:
- Average Accuracy (ACC): Mean accuracy over all tasks after final training
- Last Task Accuracy: Highest accuracy achieved during the last task's training
- Forgetting Measure (FM): How much accuracy is lost on previous tasks
- Backward Transfer (BWT): Influence of new tasks on old task performance
- Forward Transfer (FWT): Influence of old tasks on new task performance (zero-shot)

Key notation:
- R[i, j] = accuracy on task j after finishing training on task i (i >= j for standard CL)
- T = total number of tasks
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@dataclass
class AccuracyMatrix:
    """
    Tracks the accuracy matrix R[i, j] for continual learning evaluation.
    
    R[i, j] = accuracy on task j after training on task i
    
    We also track accuracy at each global round within each task for finer-grained analysis.
    
    Structure:
    - accuracy_matrix[task_i][task_j] = accuracy on task_j after finishing task_i
    - round_accuracies[task_i][round_r][task_j] = accuracy on task_j at round r of task_i
    - best_accuracies[task_j] = best accuracy ever achieved on task_j
    """
    num_tasks: int
    rounds_per_task: int
    
    # R[i, j] = accuracy on task j after finishing task i
    # Shape conceptually: (num_tasks, num_tasks), but we store as nested dict for flexibility
    accuracy_matrix: Dict[int, Dict[int, float]] = field(default_factory=dict)
    
    # Finer-grained: accuracy at each round
    # round_accuracies[task_id][round_idx][eval_task_id] = accuracy
    round_accuracies: Dict[int, Dict[int, Dict[int, float]]] = field(default_factory=dict)
    
    # Track best accuracy achieved on each task (for average accuracy computation)
    best_accuracies: Dict[int, float] = field(default_factory=dict)
    
    # Track when best accuracy was achieved (task_id, round_idx)
    best_accuracy_location: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize nested dictionaries."""
        for i in range(self.num_tasks):
            self.accuracy_matrix[i] = {}
            self.round_accuracies[i] = {}
            for r in range(self.rounds_per_task):
                self.round_accuracies[i][r] = {}
    
    def record_accuracy(
        self, 
        current_task: int, 
        current_round: int, 
        eval_task: int, 
        accuracy: float
    ) -> None:
        """
        Record accuracy on eval_task during training of current_task at current_round.
        
        Args:
            current_task: Task currently being trained (0-indexed)
            current_round: Current global round within the task (0-indexed)
            eval_task: Task being evaluated (0-indexed)
            accuracy: Test accuracy on eval_task
        """
        # Record in round_accuracies
        if current_task not in self.round_accuracies:
            self.round_accuracies[current_task] = {}
        if current_round not in self.round_accuracies[current_task]:
            self.round_accuracies[current_task][current_round] = {}
        
        self.round_accuracies[current_task][current_round][eval_task] = accuracy
        
        # Update best accuracy for this eval_task
        if eval_task not in self.best_accuracies or accuracy > self.best_accuracies[eval_task]:
            self.best_accuracies[eval_task] = accuracy
            self.best_accuracy_location[eval_task] = (current_task, current_round)
        
        # If this is the last round of current_task, update accuracy_matrix
        if current_round == self.rounds_per_task - 1:
            if current_task not in self.accuracy_matrix:
                self.accuracy_matrix[current_task] = {}
            self.accuracy_matrix[current_task][eval_task] = accuracy
    
    def record_task_end_accuracies(
        self, 
        current_task: int, 
        task_accuracies: Dict[int, float]
    ) -> None:
        """
        Record all task accuracies at the end of training a task.
        
        Args:
            current_task: Task that just finished training
            task_accuracies: Dict mapping eval_task_id -> accuracy
        """
        if current_task not in self.accuracy_matrix:
            self.accuracy_matrix[current_task] = {}
        self.accuracy_matrix[current_task].update(task_accuracies)
    
    def get_R(self, train_task: int, eval_task: int) -> Optional[float]:
        """
        Get R[train_task, eval_task] = accuracy on eval_task after finishing train_task.
        
        Returns None if not recorded.
        """
        if train_task in self.accuracy_matrix:
            return self.accuracy_matrix[train_task].get(eval_task)
        return None
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert accuracy matrix to numpy array.
        
        Returns:
            np.ndarray of shape (num_tasks, num_tasks) where entry [i, j] is R[i, j].
            Entries that haven't been recorded are set to NaN.
        """
        matrix = np.full((self.num_tasks, self.num_tasks), np.nan)
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if i in self.accuracy_matrix and j in self.accuracy_matrix[i]:
                    matrix[i, j] = self.accuracy_matrix[i][j]
        return matrix


def compute_average_accuracy(acc_matrix: AccuracyMatrix, use_best: bool = True) -> float:
    """
    Compute Average Accuracy over all tasks.
    
    If use_best=True (default): For each task, use the BEST accuracy achieved during
    that task's training or any subsequent task's training, then average.
    
    If use_best=False: Use the final accuracy on each task after all training.
    This is the standard formula: ACC = (1/T) * sum_{j=1}^{T} R[T, j]
    
    Args:
        acc_matrix: AccuracyMatrix with recorded accuracies
        use_best: If True, use best accuracy per task; if False, use final accuracy
        
    Returns:
        Average accuracy (float between 0 and 1)
    """
    T = acc_matrix.num_tasks
    
    if use_best:
        # Use best accuracy achieved on each task
        accs = []
        for j in range(T):
            if j in acc_matrix.best_accuracies:
                accs.append(acc_matrix.best_accuracies[j])
        if not accs:
            return 0.0
        return sum(accs) / len(accs)
    else:
        # Standard: use final accuracy (R[T-1, j] for all j)
        final_task = T - 1
        if final_task not in acc_matrix.accuracy_matrix:
            return 0.0
        accs = list(acc_matrix.accuracy_matrix[final_task].values())
        if not accs:
            return 0.0
        return sum(accs) / len(accs)


def compute_last_task_accuracy(acc_matrix: AccuracyMatrix, use_best: bool = True) -> float:
    """
    Compute accuracy on the last task.
    
    If use_best=True: Return the highest test accuracy achieved during last task training.
    If use_best=False: Return the final accuracy after all rounds of last task.
    
    Args:
        acc_matrix: AccuracyMatrix with recorded accuracies
        use_best: Whether to use best or final accuracy
        
    Returns:
        Last task accuracy (float between 0 and 1)
    """
    T = acc_matrix.num_tasks
    last_task = T - 1
    
    if use_best:
        # Find best accuracy on last task during its training
        best_acc = 0.0
        if last_task in acc_matrix.round_accuracies:
            for round_idx, round_accs in acc_matrix.round_accuracies[last_task].items():
                if last_task in round_accs:
                    best_acc = max(best_acc, round_accs[last_task])
        return best_acc
    else:
        # Use final accuracy
        return acc_matrix.get_R(last_task, last_task) or 0.0


def compute_forgetting_measure(acc_matrix: AccuracyMatrix) -> Tuple[float, Dict[int, float]]:
    """
    Compute Forgetting Measure (FM).
    
    For each task j < T:
        f_j = max_{i in {j, ..., T-1}} R[i, j] - R[T-1, j]
    
    This measures how much accuracy is lost on task j between its peak and the end.
    
    FM = (1 / (T-1)) * sum_{j=0}^{T-2} f_j
    
    Args:
        acc_matrix: AccuracyMatrix with recorded accuracies
        
    Returns:
        Tuple of (average forgetting, dict of per-task forgetting)
    """
    T = acc_matrix.num_tasks
    
    if T <= 1:
        return 0.0, {}
    
    per_task_forgetting: Dict[int, float] = {}
    
    for j in range(T - 1):  # For tasks 0 to T-2
        # Find max accuracy on task j across all tasks i >= j
        max_acc_j = 0.0
        for i in range(j, T):
            acc = acc_matrix.get_R(i, j)
            if acc is not None:
                max_acc_j = max(max_acc_j, acc)
        
        # Get final accuracy on task j
        final_acc_j = acc_matrix.get_R(T - 1, j) or 0.0
        
        # Forgetting for task j
        f_j = max_acc_j - final_acc_j
        per_task_forgetting[j] = f_j
    
    # Average forgetting
    if per_task_forgetting:
        avg_forgetting = sum(per_task_forgetting.values()) / len(per_task_forgetting)
    else:
        avg_forgetting = 0.0
    
    return avg_forgetting, per_task_forgetting


def compute_backward_transfer(acc_matrix: AccuracyMatrix) -> Tuple[float, Dict[int, float]]:
    """
    Compute Backward Transfer (BWT).
    
    BWT = (1 / (T-1)) * sum_{j=0}^{T-2} (R[T-1, j] - R[j, j])
    
    Measures how learning new tasks affects performance on previous tasks.
    - Negative BWT indicates forgetting (common)
    - Positive BWT indicates learning new tasks helped old tasks (rare, beneficial)
    
    Args:
        acc_matrix: AccuracyMatrix with recorded accuracies
        
    Returns:
        Tuple of (average BWT, dict of per-task BWT)
    """
    T = acc_matrix.num_tasks
    
    if T <= 1:
        return 0.0, {}
    
    per_task_bwt: Dict[int, float] = {}
    
    for j in range(T - 1):  # For tasks 0 to T-2
        # R[j, j] = accuracy on task j right after training on it
        R_jj = acc_matrix.get_R(j, j)
        # R[T-1, j] = accuracy on task j after training on all tasks
        R_Tj = acc_matrix.get_R(T - 1, j)
        
        if R_jj is not None and R_Tj is not None:
            per_task_bwt[j] = R_Tj - R_jj
    
    if per_task_bwt:
        avg_bwt = sum(per_task_bwt.values()) / len(per_task_bwt)
    else:
        avg_bwt = 0.0
    
    return avg_bwt, per_task_bwt


def compute_forward_transfer(
    acc_matrix: AccuracyMatrix, 
    random_baseline: Optional[Dict[int, float]] = None
) -> Tuple[float, Dict[int, float]]:
    """
    Compute Forward Transfer (FWT).
    
    FWT = (1 / (T-1)) * sum_{j=1}^{T-1} (R[j-1, j] - b_j)
    
    Where R[j-1, j] is the "zero-shot" accuracy on task j after training on tasks 0..j-1,
    and b_j is a random baseline (often 1/num_classes for classification).
    
    Measures how learning previous tasks helps with new tasks (before seeing their data).
    
    Note: In our setup, we evaluate on task j AFTER starting to train on it, so we
    approximate FWT using the accuracy at round 0 of task j (minimal exposure).
    
    Args:
        acc_matrix: AccuracyMatrix with recorded accuracies
        random_baseline: Optional dict mapping task_id -> random baseline accuracy.
                        If None, uses 0 as baseline.
        
    Returns:
        Tuple of (average FWT, dict of per-task FWT)
    """
    T = acc_matrix.num_tasks
    
    if T <= 1:
        return 0.0, {}
    
    per_task_fwt: Dict[int, float] = {}
    
    for j in range(1, T):  # For tasks 1 to T-1
        # Zero-shot: accuracy on task j after training only on tasks 0..j-1
        # Approximation: use R[j-1, j] if available (evaluated at end of task j-1)
        # Or use first round accuracy on task j
        
        R_prev_j = acc_matrix.get_R(j - 1, j)
        
        # If we don't have R[j-1, j], try to get accuracy from round 0 of task j
        if R_prev_j is None:
            if j in acc_matrix.round_accuracies and 0 in acc_matrix.round_accuracies[j]:
                R_prev_j = acc_matrix.round_accuracies[j][0].get(j, 0.0)
        
        if R_prev_j is None:
            R_prev_j = 0.0
        
        # Get baseline
        baseline = 0.0
        if random_baseline is not None and j in random_baseline:
            baseline = random_baseline[j]
        
        per_task_fwt[j] = R_prev_j - baseline
    
    if per_task_fwt:
        avg_fwt = sum(per_task_fwt.values()) / len(per_task_fwt)
    else:
        avg_fwt = 0.0
    
    return avg_fwt, per_task_fwt


@dataclass
class MetricsSummary:
    """Summary of all continual learning metrics."""
    
    average_accuracy: float
    average_accuracy_best: float  # Using best per-task accuracy
    last_task_accuracy: float
    last_task_accuracy_best: float
    forgetting_measure: float
    per_task_forgetting: Dict[int, float]
    backward_transfer: float
    per_task_bwt: Dict[int, float]
    forward_transfer: float
    per_task_fwt: Dict[int, float]
    accuracy_matrix: np.ndarray
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "average_accuracy": self.average_accuracy,
            "average_accuracy_best": self.average_accuracy_best,
            "last_task_accuracy": self.last_task_accuracy,
            "last_task_accuracy_best": self.last_task_accuracy_best,
            "forgetting_measure": self.forgetting_measure,
            "per_task_forgetting": self.per_task_forgetting,
            "backward_transfer": self.backward_transfer,
            "per_task_bwt": self.per_task_bwt,
            "forward_transfer": self.forward_transfer,
            "per_task_fwt": self.per_task_fwt,
            "accuracy_matrix": self.accuracy_matrix.tolist(),
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Continual Learning Metrics Summary",
            "=" * 60,
            f"Average Accuracy (final):     {self.average_accuracy:.4f}",
            f"Average Accuracy (best):      {self.average_accuracy_best:.4f}",
            f"Last Task Accuracy (final):   {self.last_task_accuracy:.4f}",
            f"Last Task Accuracy (best):    {self.last_task_accuracy_best:.4f}",
            f"Forgetting Measure:           {self.forgetting_measure:.4f}",
            f"Backward Transfer (BWT):      {self.backward_transfer:.4f}",
            f"Forward Transfer (FWT):       {self.forward_transfer:.4f}",
            "-" * 60,
            "Per-task Forgetting:",
        ]
        for task_id, f in sorted(self.per_task_forgetting.items()):
            lines.append(f"  Task {task_id}: {f:.4f}")
        lines.append("-" * 60)
        lines.append("Per-task Backward Transfer:")
        for task_id, bwt in sorted(self.per_task_bwt.items()):
            lines.append(f"  Task {task_id}: {bwt:.4f}")
        lines.append("-" * 60)
        lines.append("Accuracy Matrix R[i,j] (row=train_task, col=eval_task):")
        lines.append(str(self.accuracy_matrix))
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_all_metrics(
    acc_matrix: AccuracyMatrix,
    random_baseline: Optional[Dict[int, float]] = None
) -> MetricsSummary:
    """
    Compute all continual learning metrics from an accuracy matrix.
    
    Args:
        acc_matrix: AccuracyMatrix with recorded accuracies
        random_baseline: Optional baseline for FWT computation
        
    Returns:
        MetricsSummary containing all metrics
    """
    avg_acc_final = compute_average_accuracy(acc_matrix, use_best=False)
    avg_acc_best = compute_average_accuracy(acc_matrix, use_best=True)
    
    last_acc_final = compute_last_task_accuracy(acc_matrix, use_best=False)
    last_acc_best = compute_last_task_accuracy(acc_matrix, use_best=True)
    
    fm, per_task_fm = compute_forgetting_measure(acc_matrix)
    bwt, per_task_bwt = compute_backward_transfer(acc_matrix)
    fwt, per_task_fwt = compute_forward_transfer(acc_matrix, random_baseline)
    
    return MetricsSummary(
        average_accuracy=avg_acc_final,
        average_accuracy_best=avg_acc_best,
        last_task_accuracy=last_acc_final,
        last_task_accuracy_best=last_acc_best,
        forgetting_measure=fm,
        per_task_forgetting=per_task_fm,
        backward_transfer=bwt,
        per_task_bwt=per_task_bwt,
        forward_transfer=fwt,
        per_task_fwt=per_task_fwt,
        accuracy_matrix=acc_matrix.to_numpy(),
    )


def save_metrics(summary: MetricsSummary, output_dir: Path, filename: str = "metrics.json") -> Path:
    """
    Save metrics summary to a JSON file.
    
    Args:
        summary: MetricsSummary to save
        output_dir: Directory to save to
        filename: Name of the output file
        
    Returns:
        Path to the saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)
    
    log.info(f"Metrics saved to {output_path}")
    return output_path


def save_accuracy_matrix_csv(acc_matrix: AccuracyMatrix, output_dir: Path, filename: str = "accuracy_matrix.csv") -> Path:
    """
    Save the accuracy matrix to a CSV file for easy inspection.
    
    Args:
        acc_matrix: AccuracyMatrix to save
        output_dir: Directory to save to
        filename: Name of the output file
        
    Returns:
        Path to the saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matrix = acc_matrix.to_numpy()
    output_path = output_dir / filename
    
    # Write with headers
    with open(output_path, 'w') as f:
        # Header row
        headers = ["train_task"] + [f"eval_task_{j}" for j in range(acc_matrix.num_tasks)]
        f.write(",".join(headers) + "\n")
        
        # Data rows
        for i in range(acc_matrix.num_tasks):
            row = [str(i)] + [f"{matrix[i, j]:.4f}" if not np.isnan(matrix[i, j]) else "" 
                             for j in range(acc_matrix.num_tasks)]
            f.write(",".join(row) + "\n")
    
    log.info(f"Accuracy matrix saved to {output_path}")
    return output_path


# -------------------------
# Image Generation Utilities
# -------------------------

@torch.no_grad()
def generate_class_samples(
    generator: nn.Module,
    class_ids: List[int],
    num_samples_per_class: int = 4,
    z_dim: int = 128,
    device: torch.device = torch.device('cpu'),
) -> Dict[int, torch.Tensor]:
    """
    Generate sample images for each class using a conditional generator.
    
    Args:
        generator: Conditional generator model with forward(z, y) -> images
        class_ids: List of class IDs to generate samples for
        num_samples_per_class: Number of samples per class
        z_dim: Latent dimension for the generator
        device: Device to generate on
        
    Returns:
        Dict mapping class_id -> tensor of shape (num_samples, C, H, W)
    """
    generator.eval()
    samples = {}
    
    for class_id in class_ids:
        z = torch.randn(num_samples_per_class, z_dim, device=device)
        y = torch.full((num_samples_per_class,), class_id, dtype=torch.long, device=device)
        
        images = generator(z, y)
        samples[class_id] = images.cpu()
    
    return samples


def save_class_samples_grid(
    samples: Dict[int, torch.Tensor],
    output_dir: Path,
    filename: str = "generated_samples.png",
    nrow_per_class: int = 4,
) -> Path:
    """
    Save generated samples as a grid image.
    
    Creates one row per class, with samples arranged horizontally.
    
    Args:
        samples: Dict mapping class_id -> tensor of shape (num_samples, C, H, W)
        output_dir: Directory to save the image
        filename: Output filename
        nrow_per_class: Number of samples per row for each class
        
    Returns:
        Path to the saved image
    """
    try:
        import torchvision.utils as vutils
        from PIL import Image
    except ImportError:
        log.warning("torchvision or PIL not available, skipping image generation")
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort classes for consistent ordering
    sorted_classes = sorted(samples.keys())
    
    # Concatenate all samples
    all_images = []
    for class_id in sorted_classes:
        class_samples = samples[class_id]
        # Take up to nrow_per_class samples
        class_samples = class_samples[:nrow_per_class]
        all_images.append(class_samples)
    
    if not all_images:
        log.warning("No samples to save")
        return None
    
    # Stack all images: each class is one row
    # Total images = num_classes * nrow_per_class
    all_images = torch.cat(all_images, dim=0)
    
    # Create grid
    num_classes = len(sorted_classes)
    grid = vutils.make_grid(all_images, nrow=nrow_per_class, normalize=True, padding=2)
    
    # Convert to PIL and save
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
    
    # Handle grayscale
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
    
    img = Image.fromarray(grid_np)
    output_path = output_dir / filename
    img.save(output_path)
    
    log.info(f"Generated samples saved to {output_path}")
    return output_path


def generate_and_save_samples(
    generator: nn.Module,
    known_classes: List[int],
    output_dir: Path,
    num_samples_per_class: int = 4,
    z_dim: int = 128,
    device: torch.device = torch.device('cpu'),
    filename: str = "generated_samples.png",
) -> Optional[Path]:
    """
    Generate and save sample images for all known classes.
    
    Creates a grid with one row per class, each row containing num_samples_per_class images.
    
    Args:
        generator: Conditional generator model
        known_classes: List of class IDs that have been learned
        output_dir: Directory to save the image
        num_samples_per_class: Number of samples per class (columns)
        z_dim: Latent dimension for the generator
        device: Device to generate on
        filename: Output filename
        
    Returns:
        Path to the saved image, or None if generation failed
    """
    if not known_classes:
        log.warning("No known classes to generate samples for")
        return None
    
    try:
        samples = generate_class_samples(
            generator=generator,
            class_ids=known_classes,
            num_samples_per_class=num_samples_per_class,
            z_dim=z_dim,
            device=device,
        )
        
        return save_class_samples_grid(
            samples=samples,
            output_dir=output_dir,
            filename=filename,
            nrow_per_class=num_samples_per_class,
        )
    except Exception as e:
        log.error(f"Failed to generate samples: {e}")
        return None


# -------------------------
# Evaluation Helpers
# -------------------------

@torch.no_grad()
def evaluate_on_task(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device
) -> float:
    """
    Evaluate model accuracy on a single task's test data.
    
    Args:
        model: Model to evaluate
        loader: DataLoader for the task's test data
        device: Device to evaluate on
        
    Returns:
        Accuracy (float between 0 and 1)
    """
    model.eval()
    correct, total = 0, 0
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    
    return float(correct) / max(total, 1)


def evaluate_all_tasks(
    model: nn.Module,
    task_loaders: Dict[int, DataLoader],
    device: torch.device,
) -> Dict[int, float]:
    """
    Evaluate model on all available task test sets.
    
    Args:
        model: Model to evaluate
        task_loaders: Dict mapping task_id -> test DataLoader
        device: Device to evaluate on
        
    Returns:
        Dict mapping task_id -> accuracy
    """
    results = {}
    for task_id, loader in task_loaders.items():
        results[task_id] = evaluate_on_task(model, loader, device)
    return results
