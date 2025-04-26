import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

# Parse arguments
parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on ASL dataset")
parser.add_argument("--base_dir", type=str, default="asl_data", help="Base directory with data")
parser.add_argument("--output_dir", type=str, default="asl_diffusion", help="Output directory")
parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Pretrained model ID")
parser.add_argument("--resolution", type=int, default=512, help="Image resolution for training")
parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
parser.add_argument("--save_every_n_epochs", type=int, default=1, help="Save checkpoint every n epochs")
args = parser.parse_args()

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
    
    # Load pipeline, VAE and UNet
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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Move models to device
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)
    
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
        num_workers=2,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(device)
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
                input_ids = batch["input_ids"].to(device)
                encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Predict the noise residual
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            train_loss += loss.detach().item()
            
            logs = {"train_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            global_step += 1
        
        # Calculate average training loss
        train_loss = train_loss / len(train_dataloader)
        
        # Validation loop
        unet.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(device)
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
                input_ids = batch["input_ids"].to(device)
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
                val_loss += loss.detach().item()
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_dataloader)
        
        # Print progress
        print(f"Epoch {epoch}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            # Save the fine-tuned pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model,
                unet=unet,
            )
            
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            pipeline.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
            
            # Generate a sample image from the validation set
            pipeline = pipeline.to(device)
            
            # Get a sample caption
            sample_caption = None
            for batch in val_dataloader:
                sample_caption = batch["caption"][0]
                break
            
            if sample_caption:
                print(f"Generating sample image for: {sample_caption}")
                image = pipeline(sample_caption, num_inference_steps=30).images[0]
                image_path = os.path.join(checkpoint_dir, "sample.png")
                image.save(image_path)
                print(f"Sample image saved to {image_path}")
    
    # Save the final model
    # Save the fine-tuned pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        unet=unet,
    )
    
    final_dir = os.path.join(args.output_dir, "final")
    pipeline.save_pretrained(final_dir)
    print(f"Saved final model to {final_dir}")

if __name__ == "__main__":
    main()
