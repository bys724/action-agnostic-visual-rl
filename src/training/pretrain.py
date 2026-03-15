"""
Training utilities for video prediction models.

Provides training loop, evaluation, and checkpoint management
for TwoStreamModel, SingleStreamModel, and VideoMAEModel.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def train_epoch(model, dataloader, optimizer, device, epoch, dataset=None):
    """
    Train for one epoch with multi-gap weighted loss.

    Args:
        model: Video prediction model
        dataloader: DataLoader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
        dataset: Dataset with get_loss_weight() method (optional)

    Returns:
        avg_loss: Average unweighted loss for the epoch
    """
    model.train()
    total_loss = 0
    total_weighted_loss = 0
    num_batches = 0
    gap_counts = {}

    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch (img_t, img_tk, gap) or (img_t, img_tk)
        if len(batch) == 3:
            img_t, img_tk, gaps = batch
            gaps = gaps.numpy()  # [B]
        else:
            img_t, img_tk = batch
            gaps = np.ones(img_t.shape[0])  # Default gap=1

        img_t = img_t.to(device)
        img_tk = img_tk.to(device)

        optimizer.zero_grad()

        # Compute loss based on model type
        # VideoMAE: masked reconstruction (no gap weighting)
        # Two-stream/Single-stream: future prediction with gap weighting
        actual_model = model.module if hasattr(model, 'module') else model
        model_name = type(actual_model).__name__

        if model_name == 'VideoMAEModel':
            # VideoMAE: masked reconstruction loss (already scalar)
            loss, img_pred = actual_model.compute_loss(img_t, img_tk)
            weighted_loss = loss
            unweighted_loss = loss
        else:
            # Two-stream/Single-stream: future prediction with gap weighting
            img_pred, _ = model(img_t, img_tk)
            per_sample_loss = F.mse_loss(img_pred, img_tk, reduction='none')
            per_sample_loss = per_sample_loss.mean(dim=(1, 2, 3))  # [B]

            # Apply gap-dependent weights
            if dataset is not None and hasattr(dataset, 'get_loss_weight'):
                weights = torch.tensor(
                    [dataset.get_loss_weight(int(g)) for g in gaps],
                    device=device, dtype=per_sample_loss.dtype
                )
            else:
                weights = torch.ones_like(per_sample_loss)

            # Weighted loss
            weighted_loss = (per_sample_loss * weights).mean()
            unweighted_loss = per_sample_loss.mean()

        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += unweighted_loss.item()
        total_weighted_loss += weighted_loss.item()
        num_batches += 1

        # Track gap distribution
        for g in gaps:
            gap_counts[int(g)] = gap_counts.get(int(g), 0) + 1

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {unweighted_loss.item():.4f}, "
                  f"Weighted: {weighted_loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_weighted = total_weighted_loss / num_batches

    # Print gap distribution
    total_samples = sum(gap_counts.values())
    gap_dist = {k: f"{v/total_samples*100:.1f}%" for k, v in sorted(gap_counts.items())}
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Weighted = {avg_weighted:.4f}")
    print(f"  Gap distribution: {gap_dist}")

    return avg_loss


def evaluate(model, eval_dataset, device, batch_size=8, num_samples=500):
    """
    Evaluate model on validation/test dataset.

    Args:
        model: Video prediction model (or DataParallel wrapped)
        eval_dataset: Dataset with __getitem__ returning (img_t, img_tk, gap)
        device: Device
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate (for speed)

    Returns:
        dict: {'loss': float, 'weighted_loss': float, 'gap_distribution': dict}
    """
    model.eval()

    # Limit sample size
    eval_size = min(num_samples, len(eval_dataset))
    indices = np.random.choice(len(eval_dataset), eval_size, replace=False)

    total_loss = 0
    total_weighted_loss = 0
    num_batches = 0
    gap_counts = {}

    # Manual batch processing (for fast evaluation)
    for i in range(0, eval_size, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = [eval_dataset[idx] for idx in batch_indices]

        img_t = torch.stack([d[0] for d in batch_data]).to(device)
        img_tk = torch.stack([d[1] for d in batch_data]).to(device)
        gaps = np.array([d[2] for d in batch_data])

        # Forward (handle VideoMAE vs prediction models)
        actual_model = model.module if hasattr(model, 'module') else model
        model_name = type(actual_model).__name__

        with torch.no_grad():
            if model_name == 'VideoMAEModel':
                # VideoMAE: masked reconstruction
                loss, img_pred = actual_model.compute_loss(img_t, img_tk)
                weighted_loss = loss
                unweighted_loss = loss
            else:
                # Two-stream/Single-stream: future prediction
                img_pred, _ = model(img_t, img_tk)
                per_sample_loss = F.mse_loss(img_pred, img_tk, reduction='none')
                per_sample_loss = per_sample_loss.mean(dim=(1, 2, 3))

                # Gap-dependent weights
                if hasattr(eval_dataset, 'get_loss_weight'):
                    weights = torch.tensor(
                        [eval_dataset.get_loss_weight(int(g)) for g in gaps],
                        device=device, dtype=per_sample_loss.dtype
                    )
                else:
                    weights = torch.ones_like(per_sample_loss)

                weighted_loss = (per_sample_loss * weights).mean()
                unweighted_loss = per_sample_loss.mean()

        total_loss += unweighted_loss.item()
        total_weighted_loss += weighted_loss.item()
        num_batches += 1

        for g in gaps:
            gap_counts[int(g)] = gap_counts.get(int(g), 0) + 1

    avg_loss = total_loss / num_batches
    avg_weighted = total_weighted_loss / num_batches

    model.train()

    return {
        'loss': avg_loss,
        'weighted_loss': avg_weighted,
        'gap_distribution': gap_counts,
    }


def train(
    model,
    train_dataset,
    num_epochs=10,
    batch_size=8,
    lr=1e-4,
    device="cuda",
    checkpoint_dir=None,
    save_interval=None,
    eval_dataset=None,
    eval_interval=1,
    resume_from=None,
    multi_gpu=True,
):
    """
    Main training loop with periodic evaluation and checkpointing.

    Args:
        model: Video prediction model
        train_dataset: Training dataset
        num_epochs: Number of training epochs
        batch_size: Batch size (per GPU if multi_gpu=True)
        lr: Learning rate
        device: Device to train on
        checkpoint_dir: Base directory for checkpoints (auto-creates timestamped subfolder)
        save_interval: Save checkpoint every N epochs (None = only save best)
        eval_dataset: Evaluation dataset (optional)
        eval_interval: Evaluate every N epochs
        resume_from: Path to checkpoint to resume training from
        multi_gpu: Use DataParallel if multiple GPUs available

    Returns:
        model: Trained model
        history: Training history dict
    """
    # Multi-GPU setup
    use_multi_gpu = multi_gpu and torch.cuda.device_count() > 1
    if use_multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        # Adjust batch size for multi-GPU (batch_size is per-GPU)
        effective_batch_size = batch_size * torch.cuda.device_count()
        print(f"  Per-GPU batch size: {batch_size}")
        print(f"  Effective batch size: {effective_batch_size}")
    else:
        model = model.to(device)
        effective_batch_size = batch_size

    # Setup checkpoint directory with timestamp
    run_dir = None
    if checkpoint_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(checkpoint_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint directory: {run_dir}")

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=32 if use_multi_gpu else 16,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_eval_loss = float('inf')
    history = {
        'train_loss': [],
        'eval_loss': [],
        'epoch_time': [],        # Seconds per epoch
        'samples_per_sec': [],   # Training throughput
        'timestamps': [],        # ISO timestamp at epoch end
    }

    if resume_from:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        # Handle DataParallel: load into model.module if wrapped
        model_to_load = model.module if use_multi_gpu else model
        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"  WARNING: Checkpoint incompatible — {e}")
            print(f"  Starting from scratch (ignoring old checkpoint)")
            resume_from = None
        if resume_from:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'history' in checkpoint:
                history = checkpoint['history']
                # Ensure new metrics exist in resumed history
                history.setdefault('epoch_time', [])
                history.setdefault('samples_per_sec', [])
                history.setdefault('timestamps', [])
            if 'best_eval_loss' in checkpoint:
                best_eval_loss = checkpoint['best_eval_loss']
            print(f"  Resumed from epoch {checkpoint['epoch']}, LR: {scheduler.get_last_lr()[0]:.2e}")

    print(f"\nTraining for {num_epochs} epochs (starting from epoch {start_epoch})")
    print(f"  Train dataset: {len(train_dataset)} samples")
    if eval_dataset:
        print(f"  Eval dataset: {len(eval_dataset)} samples (every {eval_interval} epochs)")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    if save_interval:
        print(f"  Save interval: every {save_interval} epochs")
    print()

    # TensorBoard
    writer = None
    if run_dir and HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=str(run_dir / 'tb'))
        print(f"TensorBoard: {run_dir / 'tb'}")

    # Save config
    if run_dir:
        config = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'eval_interval': eval_interval,
            'save_interval': save_interval,
            'train_dataset_size': len(train_dataset),
            'eval_dataset_size': len(eval_dataset) if eval_dataset else 0,
            'start_time': datetime.now().isoformat(),
        }
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()

        # Train
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch, dataset=train_dataset)
        scheduler.step()
        history['train_loss'].append(avg_loss)

        # Evaluate
        eval_loss = None
        if eval_dataset and epoch % eval_interval == 0:
            eval_result = evaluate(model, eval_dataset, device, batch_size=batch_size)
            eval_loss = eval_result['loss']
            history['eval_loss'].append(eval_loss)
            print(f"  [Eval] Loss: {eval_loss:.4f}, Weighted: {eval_result['weighted_loss']:.4f}")

        # Track time metrics
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        samples_per_sec = len(train_dataset) / epoch_duration
        current_timestamp = datetime.now().isoformat()

        history['epoch_time'].append(epoch_duration)
        history['samples_per_sec'].append(samples_per_sec)
        history['timestamps'].append(current_timestamp)

        # Print time metrics
        current_lr = scheduler.get_last_lr()[0]
        print(f"  [Time] Epoch: {epoch_duration:.1f}s, Throughput: {samples_per_sec:.1f} samples/sec, LR: {current_lr:.2e}")

        # Estimate remaining time
        if epoch < num_epochs:
            avg_epoch_time = sum(history['epoch_time']) / len(history['epoch_time'])
            remaining_epochs = num_epochs - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_hours = eta_seconds / 3600
            print(f"  [ETA] {remaining_epochs} epochs remaining, ~{eta_hours:.1f}h ({eta_seconds/60:.0f}min)")

        # TensorBoard logging
        if writer:
            writer.add_scalar('loss/train', avg_loss, epoch)
            if eval_loss is not None:
                writer.add_scalar('loss/eval', eval_loss, epoch)
            writer.add_scalar('lr', current_lr, epoch)
            writer.add_scalar('perf/samples_per_sec', samples_per_sec, epoch)
            writer.add_scalar('perf/epoch_time_sec', epoch_duration, epoch)
            writer.flush()

        # Save checkpoint
        if run_dir:
            # Handle DataParallel: save model.module.state_dict() if wrapped
            model_to_save = model.module if use_multi_gpu else model
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'eval_loss': eval_loss,
                'best_eval_loss': best_eval_loss,
                'history': history,
            }

            # Save periodic checkpoint
            if save_interval and epoch % save_interval == 0:
                ckpt_path = run_dir / f'checkpoint_epoch{epoch:04d}.pt'
                torch.save(checkpoint_data, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path.name}")

            # Save best model
            current_loss = eval_loss if eval_loss is not None else avg_loss
            if current_loss < best_eval_loss:
                best_eval_loss = current_loss
                checkpoint_data['best_eval_loss'] = best_eval_loss
                best_path = run_dir / 'best_model.pt'
                torch.save(checkpoint_data, best_path)
                print(f"  Saved best model (loss: {current_loss:.4f})")

            # Always save latest (for resume)
            latest_path = run_dir / 'latest.pt'
            torch.save(checkpoint_data, latest_path)

    # Save final history
    if run_dir:
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    if writer:
        writer.close()

    return model, history
