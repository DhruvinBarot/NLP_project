import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on ASL dataset")
parser.add_argument("--base_dir", type=str, default="asl_data", help="Base directory with data")
parser.add_argument("--output_dir", type=str, default="asl_diffusion", help="Output directory")
parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Pretrained model ID")
parser.add_argument("--resolution", type=int, default=512, help="Image resolution for training")
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type")
parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of warm-up steps for LR scheduler")
parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision mode")
parser.add_argument("--save_every_n_epochs", type=int, default=1, help="Save checkpoint every n epochs")
parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
args = parser.parse_args()

# Set up accelerator
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    log_with="wandb" if args.use_wandb else None,
)

if accelerator.is_main_process and args.use_wandb:
    # Initialize wandb
    wandb.init(project="asl-diffusion", name="fine-tuning")

# ASL Dataset class
class ASLDataset(Dataset):
    def __init__(self, data_dir, tokenizer, resolution=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Get all annotation files
        self.xml_files = [f for f in os.listdir(data_dir) if f.endswith('.xml')]
        print(f"Found {len(self.xml_files)} samples in {data_dir}")
        
    def __len__(self):
        return len(self.xml_files)
    
    def __getitem__(self, idx):
        xml_file = self.xml_files[idx]
        xml_path = os.path.join(self.data_dir, xml_file)
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image filename and load image
        img_filename = root.find('.//filename').text
        img_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        
        # Extract class names (ASL signs)
        classes = []
        for obj in root.findall('.//object'):
            class_name = obj.find('name').text
            classes.append(class_name)
        
        # Create caption: "ASL sign for [class]"
        if len(classes) == 1:
            caption = f"ASL sign for {classes[0]}"
        else:
            # Multiple classes in one image
            unique_classes = list(set(classes))
            caption = f"ASL signs for {', '.join(unique_classes[:-1])} and {unique_classes[-1]}"
        
        # Encode text
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_inputs.input_ids[0]
        
        # Transform image
        pixel_values = self.transform(image)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_embeddings,
            "caption": caption,
        }

# Load models
def load_models(pretrained_model_path):
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )
    
    # Load VAE and UNet
    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_path)
    vae = pipeline.vae
    unet = pipeline.unet
    
    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler"
    )
    
    return tokenizer, text_encoder, vae, unet, noise_scheduler

def main():
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data directories
    train_dir = os.path.join(args.base_dir, "splits", "train")
    val_dir = os.path.join(args.base_dir, "splits", "val")
    
    # Check if split directories exist
    if not os.path.exists(train_dir):
        print("Training directory not found. Using VOC directory instead.")
        train_dir = os.path.join(args.base_dir, "voc")
    if not os.path.exists(val_dir):
        print("Validation directory not found. Using 10% of training data for validation.")
        val_dir = train_dir
    
    # Load models
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(args.pretrained_model)
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Create datasets and dataloaders
    train_dataset = ASLDataset(train_dir, tokenizer, args.resolution)
    
    # If val_dir is the same as train_dir, split the train dataset
    if val_dir == train_dir:
        # Use 90% for training, 10% for validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    else:
        val_dataset = ASLDataset(val_dir, tokenizer, args.resolution)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * args.num_train_epochs,
    )
    
    # Prepare for training with accelerator
    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Set text encoder and VAE to eval mode
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.eval()
    vae.eval()
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(accelerator.device)
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                
                # Sample noise
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device
                ).long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embeddings
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(accelerator.device)
                    encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                
                # Update parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.update(1)
                train_loss += loss.detach().item()
                
                logs = {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                global_step += 1
                
                # Log to wandb
                if accelerator.is_main_process and args.use_wandb:
                    accelerator.log(logs, step=global_step)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_dataloader)
        
        # Validation loop
        unet.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, disable=not accelerator.is_local_main_process, desc="Validation"):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                
                # Sample noise
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device
                ).long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embeddings
                input_ids = batch["input_ids"].to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
                val_loss += loss.detach().item()
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_dataloader)
        
        # Log validation loss
        logs = {"val_loss": val_loss, "train_loss": train_loss, "epoch": epoch}
        accelerator.print(f"Epoch {epoch}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        
        if accelerator.is_main_process and args.use_wandb:
            accelerator.log(logs, step=global_step)
        
        # Save model checkpoint
        if accelerator.is_main_process and (epoch + 1) % args.save_every_n_epochs == 0:
            # Unwrap the model
            unwrapped_unet = accelerator.unwrap_model(unet)
            
            # Save the fine-tuned pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model,
                unet=unwrapped_unet,
                torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
            )
            
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            pipeline.save_pretrained(checkpoint_dir)
            accelerator.print(f"Saved checkpoint to {checkpoint_dir}")
            
            # Generate sample images
            if accelerator.is_main_process:
                pipeline = pipeline.to(accelerator.device)
                
                # Sample some captions from the validation set
                sample_captions = []
                for i, batch in enumerate(val_dataloader):
                    sample_captions.extend(batch["caption"])
                    if len(sample_captions) >= 4:
                        sample_captions = sample_captions[:4]
                        break
                
                # Generate images for each caption
                for i, caption in enumerate(sample_captions):
                    image = pipeline(caption, num_inference_steps=30).images[0]
                    image_path = os.path.join(checkpoint_dir, f"sample_{i}.png")
                    image.save(image_path)
                    
                    if args.use_wandb:
                        wandb.log({f"sample_{i}": wandb.Image(np.array(image), caption=caption)})
    
    # Save the final model
    if accelerator.is_main_process:
        # Unwrap the model
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # Save the fine-tuned pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model,
            unet=unwrapped_unet,
            torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
        )
        
        final_dir = os.path.join(args.output_dir, "final")
        pipeline.save_pretrained(final_dir)
        accelerator.print(f"Saved final model to {final_dir}")
    
    # End wandb run
    if accelerator.is_main_process and args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
