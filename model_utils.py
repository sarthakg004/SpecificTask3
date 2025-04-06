import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import glob
import re
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
import lpips



class EMA:
    """Exponential Moving Average for model parameters with dynamic decay"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': self.shadow,
            'backup': self.backup
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']
        


import torch
import matplotlib.pyplot as plt
import torchvision
import os
from diffusers import DDPMScheduler



def visualize_noise_schedule(train_loader, noise_scheduler, device="cuda", num_steps=10):
    temp_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=0  # Critical: no multiprocessing
    )

    batch = next(iter(temp_loader))
    clean_images = batch['latent'].to(device)

    # We'll use 4 images for visualization
    clean_images = clean_images[:4]

    noise = torch.randn_like(clean_images)

    allimgs = clean_images.clone()

    timestep_values = list(range(200, 1001, 200))  # [200, 400, 600, 800, 1000]

    for step in timestep_values:
        timesteps = torch.tensor([step-1] * clean_images.shape[0]).long().to(device)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        allimgs = torch.cat((allimgs, noisy_images))

    # Normalize for visualization
    allimgs = (allimgs - allimgs.min()) / (allimgs.max() - allimgs.min())

    grid = torchvision.utils.make_grid(allimgs, nrow=4, padding=2)

    # Plot
    plt.figure(figsize=(15, 10), dpi=300)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")

    # Add labels for clarity
    timestep_labels = ["Original"] + [f"t={t}" for t in timestep_values]
    plt.title("Progressive Noise Addition in Latent Space", fontsize=16)

    # Create a custom legend
    for i, label in enumerate(timestep_labels):
        plt.text(-15, i*clean_images.shape[2]*1.03 + clean_images.shape[2]/2,
                 label, fontsize=12, ha='right', va='center')

    plt.tight_layout()
    plt.savefig("noise_schedule_visualization.png")
    plt.show()

    return grid


class LightweightTextConditionedLDM:
    def __init__(self,
                 unet_config=None,
                 noise_scheduler_config=None,
                 output_dir="./ldm_output",
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device
        print(f"Using device: {self.device}")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.unet_config = unet_config

        # Initialize UNet model for denoising
        self.unet = UNet2DConditionModel(**self.unet_config)
        self.unet.to(self.device)

        # Initialize EMA model
        self.ema = EMA(self.unet, decay=0.9999)

        # Initialize perceptual loss with LPIPS
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(self.device)

        # Default noise scheduler configuration
        if noise_scheduler_config is None:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=False,
                prediction_type="v_prediction"
            )
        else:
            self.noise_scheduler = DDPMScheduler(**noise_scheduler_config)

        # For inference
        self.inference_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)

        # Initialize history dictionary to track metrics
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }

        # Initialize best model tracking
        self.best_val_loss = float('inf')
        self.best_val_epoch = -1

        # Initialize checkpoint tracking
        self.last_checkpoint_epoch = -1

        # Create subdirectories
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.load_training_history()

    def load_training_history(self):
        """Load existing training history if available"""
        history_path = f"{self.output_dir}/training_history.json"
        csv_path = f"{self.output_dir}/training_history.csv"

        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.history = json.load(f)
                print(f"Loaded training history: {len(self.history['epochs'])} epochs completed")

                # If JSON exists but CSV doesn't, create the CSV
                if not os.path.exists(csv_path):
                    self.save_history_to_csv()

                # Restore best validation tracking if available
                if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
                    min_val_index = np.argmin(self.history['val_loss'])
                    self.best_val_loss = self.history['val_loss'][min_val_index]
                    # Calculate the corresponding epoch
                    val_epoch_indices = list(range(0, len(self.history['epochs']),
                                             max(1, len(self.history['epochs']) // max(1, len(self.history['val_loss'])))))
                    if min_val_index < len(val_epoch_indices):
                        self.best_val_epoch = self.history['epochs'][val_epoch_indices[min_val_index]] - 1
                    print(f"Best validation loss: {self.best_val_loss} at epoch {self.best_val_epoch+1}")
            except Exception as e:
                print(f"Error loading training history: {e}")

    def find_latest_checkpoint(self):
        """Find the latest model checkpoint in the output directory"""
        checkpoint_files = glob.glob(f"{self.output_dir}/unet_epoch_*.pt")
        if not checkpoint_files:
            return None, -1

        # Extract epoch numbers from filenames
        epoch_numbers = []
        for checkpoint in checkpoint_files:
            match = re.search(r'unet_epoch_(\d+)\.pt', checkpoint)
            if match:
                epoch_numbers.append((int(match.group(1)), checkpoint))

        if not epoch_numbers:
            return None, -1

        # Find the latest epoch
        latest_epoch, latest_checkpoint = max(epoch_numbers, key=lambda x: x[0])
        return latest_checkpoint, latest_epoch - 1  # Convert to 0-indexed

    def compute_enhanced_loss(self, pred, target, noisy_latents, clean_latents, timesteps, weight_schedule=None):
        """
        Enhanced loss function combining MSE with L1 loss and dynamic timestep weighting
        """
        # Standard MSE loss (keep dimensions for weighting)
        mse_loss = F.mse_loss(pred, target, reduction='none')
        
        # Dynamic timestep weighting - focus more on difficult timesteps
        # Convert timesteps to float and normalize to [0,1] range
        t_normalized = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
        
        # Create non-linear weights that focus on mid-range timesteps
        # This creates a curve that peaks in the middle timesteps (where learning is most valuable)
        timestep_weights = 1.0 - (2.0 * t_normalized - 1.0).abs()  # Triangle function peaking at t=0.5
        timestep_weights = timestep_weights.view(-1, 1, 1, 1)  # Match dimensions for broadcasting
        
        # Apply weights to MSE loss and take mean
        weighted_mse = (mse_loss * timestep_weights).mean()
        
        # Add L1 loss component for sharper details (with small weight)
        l1_loss = F.l1_loss(pred, target)
        l1_weight = 0.1  # Small weight for L1 component
        
        combined_loss = weighted_mse + l1_weight * l1_loss
        
        return combined_loss

    def train(self,
          train_loader,
          val_loader,
          num_epochs=200,
          learning_rate=1e-4,
          weight_decay=1e-2,
          gradient_accumulation_steps=4,
          save_model_epochs=10,
          mixed_precision="fp16",
          validation_epochs=1,
          resume_training=True,
          ema_decay_start=0.9990,
          ema_decay_end=0.9999):

        # Move model to GPU explicitly
        self.unet = self.unet.to(self.device)
        start_epoch = 0

        # Check for existing checkpoints and resume training if requested
        if resume_training:
            checkpoint_path = f"{self.output_dir}/checkpoint_latest.pt"

            if os.path.exists(checkpoint_path):
                print(f"Found full checkpoint, resuming training...")
                # Load directly to GPU
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.unet.load_state_dict(checkpoint['model'])

                # Load EMA if available
                if 'ema' in checkpoint:
                    self.ema.load_state_dict(checkpoint['ema'])
                else:
                    # Initialize EMA from current model
                    self.ema = EMA(self.unet, decay=ema_decay_start)

                start_epoch = checkpoint['epoch'] + 1
                self.last_checkpoint_epoch = checkpoint['epoch']

                # Verify training history matches the checkpoint
                if self.history['epochs'] and self.history['epochs'][-1] >= start_epoch:
                    print(f"Training history consistent with checkpoint")
                else:
                    print(f"Warning: Training history may be inconsistent with checkpoint")
            else:
                latest_checkpoint, latest_epoch = self.find_latest_checkpoint()
                if latest_checkpoint:
                    print(f"Found model checkpoint at epoch {latest_epoch+1}, resuming training...")
                    self.load_model(latest_checkpoint)
                    # Initialize EMA
                    self.ema = EMA(self.unet, decay=ema_decay_start)
                    start_epoch = latest_epoch + 1
                    self.last_checkpoint_epoch = latest_epoch
        else:
            # Initialize EMA from scratch
            self.ema = EMA(self.unet, decay=ema_decay_start)

        # Initialize mixed precision training
        if mixed_precision == "fp16" and torch.cuda.is_available():
            print("Using mixed precision training (FP16)")
            scaler = torch.cuda.amp.GradScaler()
        else:
            print(f"Using full precision training ({mixed_precision})")
            scaler = None

        # Initialize optimizer with improved parameters
        optimizer = AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Calculate total training steps
        total_training_steps = (len(train_loader) * (num_epochs - start_epoch)) // gradient_accumulation_steps

        # Use OneCycleLR scheduler for better convergence
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,  # Peak at 10x the base learning rate
            total_steps=total_training_steps,
            pct_start=0.1,               # Warm up for first 10% of training
            div_factor=10,               # Start with lr/10
            final_div_factor=100,        # End with lr/1000
            anneal_strategy='cos'        # Use cosine annealing
        )

        # If resuming, fast-forward the scheduler to the correct point
        if start_epoch > 0:
            steps_to_skip = (len(train_loader) * start_epoch) // gradient_accumulation_steps
            print(f"Fast-forwarding scheduler by {steps_to_skip} steps")
            for _ in range(steps_to_skip):
                lr_scheduler.step()

        # Training loop
        global_step = start_epoch * len(train_loader) // gradient_accumulation_steps

        # Setup epoch progress bar
        epoch_progress = tqdm(
            range(start_epoch, num_epochs),
            desc="Training Progress",
            position=0,
            leave=True
        )

        # Pre-allocate buffers for commonly used tensors
        noise_buffer = None
        timesteps_buffer = None

        for epoch in epoch_progress:
            self.unet.train()
            running_loss = 0.0

            # Single progress bar for the current epoch
            progress_bar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False
            )

            # Process data in pin memory mode by default
            for step, batch in enumerate(train_loader):
                # Get the latents and text embeddings - move to GPU in one operation
                latents = batch['latent'].to(self.device, non_blocking=True)
                text_embeddings = batch['text_embedding'].to(self.device, non_blocking=True)
                batch_size = latents.shape[0]

                # Reuse or initialize noise buffer to avoid repeated allocations
                if noise_buffer is None or noise_buffer.shape != latents.shape:
                    noise_buffer = torch.randn_like(latents)
                else:
                    noise_buffer.normal_()

                # Reuse or initialize timesteps buffer
                if timesteps_buffer is None or timesteps_buffer.shape[0] != batch_size:
                    timesteps_buffer = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (batch_size,), device=self.device
                    ).long()
                else:
                    timesteps_buffer.random_(0, self.noise_scheduler.config.num_train_timesteps)

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = self.noise_scheduler.add_noise(latents, noise_buffer, timesteps_buffer)

                # Forward pass with mixed precision
                if mixed_precision == "fp16" and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        # Get the model prediction for the noise
                        noise_pred = self.unet(noisy_latents, timesteps_buffer, text_embeddings).sample

                        # Calculate the enhanced loss
                        loss = self.compute_enhanced_loss(
                            noise_pred, noise_buffer, noisy_latents, latents, timesteps_buffer,
                            weight_schedule=True
                        )
                else:
                    # Get the model prediction for the noise
                    noise_pred = self.unet(noisy_latents, timesteps_buffer, text_embeddings).sample

                    # Calculate the enhanced loss
                    loss = self.compute_enhanced_loss(
                        noise_pred, noise_buffer, noisy_latents, latents, timesteps_buffer,
                        weight_schedule=True
                    )

                # Divide the loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                running_loss += loss.item() * gradient_accumulation_steps

                # Backpropagate
                if mixed_precision == "fp16" and torch.cuda.is_available():
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)  # Memory optimization
                        lr_scheduler.step()

                        # Calculate dynamic EMA decay
                        current_step = epoch * len(train_loader) + step
                        total_steps = num_epochs * len(train_loader)
                        current_ema_decay = ema_decay_start + (ema_decay_end - ema_decay_start) * min(1.0, current_step / total_steps)
                        
                        # Update EMA decay and model
                        self.ema.decay = current_ema_decay
                        self.ema.update()
                        
                        # Periodically log the current EMA decay
                        if step % 500 == 0:
                            print(f"Current EMA decay: {current_ema_decay:.6f}")
                else:
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)  # Memory optimization
                        lr_scheduler.step()

                        # Calculate dynamic EMA decay
                        current_step = epoch * len(train_loader) + step
                        total_steps = num_epochs * len(train_loader)
                        current_ema_decay = ema_decay_start + (ema_decay_end - ema_decay_start) * min(1.0, current_step / total_steps)
                        
                        # Update EMA decay and model
                        self.ema.decay = current_ema_decay
                        self.ema.update()
                        
                        # Periodically log the current EMA decay
                        if step % 500 == 0:
                            print(f"Current EMA decay: {current_ema_decay:.6f}")

                # Update progress bar
                current_lr = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.6f}",
                    "lr": f"{current_lr:.7f}"
                })
                progress_bar.update(1)

                global_step += 1 / gradient_accumulation_steps

                # Save checkpoint periodically to handle unexpected interruptions
                # Every 20% of an epoch or at least every 1000 steps
                checkpoint_interval = max(len(train_loader) // 5, 1000 // gradient_accumulation_steps)
                if (step + 1) % checkpoint_interval == 0:
                    # Only save checkpoint on integer steps
                    if global_step.is_integer():
                        self.save_checkpoint(epoch, step, optimizer, lr_scheduler, scaler)

            progress_bar.close()

            # Calculate and log average loss for the epoch
            avg_loss = running_loss / len(train_loader)
            epoch_progress.set_postfix({
                "train_loss": f"{avg_loss:.6f}",
                "lr": f"{current_lr:.7f}"
            })

            # Update history
            self.history['train_loss'].append(avg_loss)
            self.history['learning_rates'].append(current_lr)
            self.history['epochs'].append(epoch+1)

            # Save history to JSON and CSV after every epoch
            self.save_history()
            self.save_history_to_csv()

            # Plot and save training curves after every epoch
            self.plot_training_progress()

            # Validation
            if (epoch + 1) % validation_epochs == 0:
                # Validate with regular model
                val_loss = self.validate(val_loader, epoch)

                # Validate with EMA model
                self.ema.apply_shadow()
                ema_val_loss = self.validate(val_loader, epoch, is_ema=True)
                self.ema.restore()

                # Store validation losses
                if len(self.history['val_loss']) < (epoch + 1) // validation_epochs:
                    self.history['val_loss'].append(val_loss)
                    if 'ema_val_loss' not in self.history:
                        self.history['ema_val_loss'] = []
                    self.history['ema_val_loss'].append(ema_val_loss)
                else:
                    # Update existing val_loss if it exists
                    self.history['val_loss'][(epoch + 1) // validation_epochs - 1] = val_loss
                    if 'ema_val_loss' not in self.history:
                        self.history['ema_val_loss'] = [ema_val_loss]
                    elif len(self.history['ema_val_loss']) < (epoch + 1) // validation_epochs:
                        self.history['ema_val_loss'].append(ema_val_loss)
                    else:
                        self.history['ema_val_loss'][(epoch + 1) // validation_epochs - 1] = ema_val_loss

                # Update best model checkpoint if current validation loss is better
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_epoch = epoch
                    print(f"New best validation loss: {val_loss:.6f}")
                    self.save_model(epoch, is_best=True)

                # Also track best EMA model
                if ema_val_loss < getattr(self, 'best_ema_val_loss', float('inf')):
                    self.best_ema_val_loss = ema_val_loss
                    self.best_ema_val_epoch = epoch
                    print(f"New best EMA validation loss: {ema_val_loss:.6f}")

                    # Save best EMA model
                    self.ema.apply_shadow()
                    self.save_model(epoch, is_best=True, is_ema=True)
                    self.ema.restore()

                # Update history files after validation
                self.save_history()
                self.save_history_to_csv()

                # Update plots with validation data
                self.plot_training_progress()

            # Save model
            if (epoch + 1) % save_model_epochs == 0:
                self.save_model(epoch)
                self.last_checkpoint_epoch = epoch

                # Also save EMA model
                self.ema.apply_shadow()
                self.save_model(epoch, is_ema=True)
                self.ema.restore()

                # Save a comprehensive checkpoint for resuming training
                self.save_checkpoint(epoch, len(train_loader)-1, optimizer, lr_scheduler, scaler)

            # Clear GPU cache at the end of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # At the end of training, save final models
        print("Training completed. Saving final models...")
        self.save_model(num_epochs-1, is_final=True)

        # Save final EMA model
        self.ema.apply_shadow()
        self.save_model(num_epochs-1, is_final=True, is_ema=True)
        self.ema.restore()
        
        # Save final comprehensive checkpoint
        self.save_checkpoint(num_epochs-1, len(train_loader)-1, optimizer, lr_scheduler, scaler, is_final=True)

    def save_checkpoint(self, epoch, step, optimizer, lr_scheduler, scaler=None, is_final=False):
        """Save checkpoint with optimizer state for resuming training"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model': self.unet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'history': self.history,
            'ema': self.ema.state_dict(),
            'best_val_loss': getattr(self, 'best_val_loss', float('inf')),
            'best_val_epoch': getattr(self, 'best_val_epoch', -1),
            'best_ema_val_loss': getattr(self, 'best_ema_val_loss', float('inf')),
            'best_ema_val_epoch': getattr(self, 'best_ema_val_epoch', -1)
        }

        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()

        # Choose appropriate checkpoint filename
        if is_final:
            checkpoint_path = f"{self.output_dir}/checkpoint_final.pt"
        else:
            checkpoint_path = f"{self.output_dir}/checkpoint_latest.pt"
            
        # Also save numbered checkpoint every 10 epochs for safety
        if (epoch + 1) % 10 == 0 or is_final:
            numbered_path = f"{self.output_dir}/checkpoint_epoch_{epoch+1}.pt"
            # Save to a temporary file first
            temp_numbered_path = f"{numbered_path}_temp.pt"
            torch.save(checkpoint, temp_numbered_path)
            os.replace(temp_numbered_path, numbered_path)
            print(f"Saved numbered checkpoint: {numbered_path}")

        # Save to a temporary file first for the main checkpoint
        temp_path = f"{checkpoint_path}_temp.pt"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def validate(self, val_loader, epoch=0, is_ema=False):
        model_type = "EMA" if is_ema else "Regular"
        self.unet.eval()
        val_loss = 0.0

        # Reuse buffers to avoid repeated allocations
        noise_buffer = None
        timesteps_buffer = None

        # Setup a single validation progress bar
        val_progress = tqdm(
            total=len(val_loader),
            desc=f"{model_type} Validation (Epoch {epoch+1})",
            leave=False
        )

        with torch.no_grad():
            for batch in val_loader:
                # Adapt to new dataset structure - move tensors to GPU efficiently
                latents = batch['latent'].to(self.device, non_blocking=True)
                text_embeddings = batch['text_embedding'].to(self.device, non_blocking=True)
                batch_size = latents.shape[0]

                # Reuse or create noise buffer
                if noise_buffer is None or noise_buffer.shape != latents.shape:
                    noise_buffer = torch.randn_like(latents)
                else:
                    noise_buffer.normal_()

                # Reuse or create timesteps buffer
                if timesteps_buffer is None or timesteps_buffer.shape[0] != batch_size:
                    timesteps_buffer = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (batch_size,), device=self.device
                    ).long()
                else:
                    timesteps_buffer.random_(0, self.noise_scheduler.config.num_train_timesteps)

                noisy_latents = self.noise_scheduler.add_noise(latents, noise_buffer, timesteps_buffer)

                # Single forward pass without intermediate transfers
                noise_pred = self.unet(noisy_latents, timesteps_buffer, text_embeddings).sample

                # Use the same enhanced loss function for validation to ensure consistency
                loss = self.compute_enhanced_loss(
                    noise_pred, noise_buffer, noisy_latents, latents, timesteps_buffer, 
                    weight_schedule=True
                )
                val_loss += loss.item()

                # Update validation progress bar
                val_progress.set_postfix({"val_loss": f"{loss.item():.6f}"})
                val_progress.update(1)

        val_progress.close()

        avg_val_loss = val_loss / len(val_loader)
        print(f"{model_type} validation loss: {avg_val_loss:.6f}")

        return avg_val_loss

    def save_model(self, epoch, is_best=False, is_final=False, is_ema=False):
        """Save model checkpoint"""
        # Determine model name based on parameters
        if is_final:
            if is_ema:
                path = f"{self.output_dir}/FINAL_EMA_model.pt"
            else:
                path = f"{self.output_dir}/FINAL_regular_model.pt"
        elif is_best:
            if is_ema:
                path = f"{self.output_dir}/min_val_ema_model.pt"
            else:
                path = f"{self.output_dir}/min_val_regular_model.pt"
        else:
            # Regular epoch checkpoint
            model_type = "ema" if is_ema else "regular"
            path = f"{self.output_dir}/unet_{model_type}_epoch_{epoch+1}.pt"

        # Save to temporary file first to avoid corruption
        temp_path = f"{path}_temp.pt"
        torch.save(self.unet.state_dict(), temp_path)
        os.replace(temp_path, path)
        print(f"Model saved at {path}")

    def load_model(self, path):
        """Load model from checkpoint"""
        if path.endswith('checkpoint_latest.pt') or path.endswith('checkpoint_final.pt') or 'checkpoint_epoch_' in path:
            # Load full checkpoint including optimizer state
            checkpoint = torch.load(path, map_location=self.device)
            self.unet.load_state_dict(checkpoint['model'])

            # Update history if available
            if 'history' in checkpoint:
                self.history = checkpoint['history']

            # Load EMA if available
            if 'ema' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema'])

            # Load best validation tracking
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
                self.best_val_epoch = checkpoint['best_val_epoch']

            if 'best_ema_val_loss' in checkpoint:
                self.best_ema_val_loss = checkpoint['best_ema_val_loss']
                self.best_ema_val_epoch = checkpoint['best_ema_val_epoch']

            print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}, step {checkpoint['step']}")
            return checkpoint['epoch']
        else:
            # Load only model weights (for regular model file)
            self.unet.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
            return None

    def save_history(self):
        """Save training history to JSON file"""
        # Save to temporary file first
        temp_path = f"{self.output_dir}/training_history_temp.json"
        with open(temp_path, 'w') as f:
            json.dump(self.history, f)

        # Replace the original file
        os.replace(temp_path, f"{self.output_dir}/training_history.json")

    def save_history_to_csv(self):
        """Save training history to CSV file"""
        # Create a DataFrame from the history
        df = pd.DataFrame({
            'epoch': self.history['epochs'],
            'train_loss': self.history['train_loss'],
            'learning_rate': self.history['learning_rates']
        })

        # Add validation loss if available (with NaN for epochs without validation)
        if self.history['val_loss']:
            # Calculate which epochs had validation
            val_epoch_indices = list(range(0, len(self.history['epochs']),
                                         max(1, len(self.history['epochs']) // max(1, len(self.history['val_loss'])))))

            # Trim to match the available validation loss entries
            val_epoch_indices = val_epoch_indices[:len(self.history['val_loss'])]

            # Get the actual epoch numbers for those indices
            val_epochs = [self.history['epochs'][i] for i in val_epoch_indices
                         if i < len(self.history['epochs'])]

            # Create a dictionary mapping epochs to val_loss
            val_loss_dict = {val_epochs[i]: self.history['val_loss'][i]
                           for i in range(min(len(val_epochs), len(self.history['val_loss'])))}

            # Map to the dataframe
            df['val_loss'] = df['epoch'].map(val_loss_dict)

            # Add EMA validation loss if available
            if 'ema_val_loss' in self.history and self.history['ema_val_loss']:
                # Create a dictionary mapping epochs to ema_val_loss
                ema_val_loss_dict = {val_epochs[i]: self.history['ema_val_loss'][i]
                                   for i in range(min(len(val_epochs), len(self.history['ema_val_loss'])))}

                # Map to the dataframe
                df['ema_val_loss'] = df['epoch'].map(ema_val_loss_dict)

        # Save to a temp file first
        temp_path = f"{self.output_dir}/training_history_temp.csv"
        df.to_csv(temp_path, index=False)

        # Replace the original file
        os.replace(temp_path, f"{self.output_dir}/training_history.csv")

    def plot_training_progress(self):
        """Plot training and validation loss as separate images with improved aesthetics"""
        plt.style.use('seaborn-v0_8-darkgrid')  # Use seaborn style for better aesthetics

        # Set up figure with improved visuals
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Plot training loss with a cleaner line (no markers)
        ax1.plot(self.history['epochs'], self.history['train_loss'],
                 color='#3498db', linewidth=2.5, label='Training Loss')

        # Plot validation losses if available
        has_validation = False
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            has_validation = True
            # Calculate which epochs had validation
            val_epoch_indices = list(range(0, len(self.history['epochs']),
                                         max(1, len(self.history['epochs']) // max(1, len(self.history['val_loss'])))))

            # Trim to match the available validation loss entries
            val_epoch_indices = val_epoch_indices[:len(self.history['val_loss'])]

            # Get the actual epoch numbers for those indices
            val_epochs = [self.history['epochs'][i] for i in val_epoch_indices
                         if i < len(self.history['epochs'])]

            # Plot validation loss with clean line
            ax1.plot(val_epochs, self.history['val_loss'],
                     color='#e74c3c', linewidth=2.5, label='Validation Loss')

            # Plot EMA validation loss if available
            if 'ema_val_loss' in self.history and len(self.history['ema_val_loss']) > 0:
                ax1.plot(val_epochs[:len(self.history['ema_val_loss'])],
                         self.history['ema_val_loss'],
                         color='#2ecc71', linewidth=2.5, label='EMA Validation Loss')

        # Set titles and labels with better fonts
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # Add legend with better placement and appearance
        if has_validation:
            ax1.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot learning rate separately with similar improved aesthetics
        plt.figure(figsize=(12, 5))
        plt.plot(self.history['epochs'], self.history['learning_rates'],
                 color='#9b59b6', linewidth=2.5)

        # Enhance the appearance
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.title('Learning Rate Schedule (OneCycleLR)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/learning_rate.png", dpi=300, bbox_inches='tight')
        plt.close()

    def resume_training_from_latest(self, train_loader, val_loader, **kwargs):
        """Convenience method to resume training from the latest checkpoint"""
        latest_checkpoint, _ = self.find_latest_checkpoint()
        if latest_checkpoint:
            self.train(train_loader, val_loader, resume_training=True, **kwargs)
        else:
            print("No checkpoint found. Starting training from scratch.")
            self.train(train_loader, val_loader, resume_training=False, **kwargs)


