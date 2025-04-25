import requests
import json
import torch
import base64
from diffusers import DiffusionPipeline
from IPython.display import Image, display
from PIL import Image as PILImage
import imageio
import numpy as np
import os
from tqdm import tqdm

def generate_asl_gif(text_input, output_path="asl_output.gif", frames=24, duration=5):
    """
    Generate an ASL gif from text input using a sign language diffusion model.
    
    Args:
        text_input (str): The text to be translated into ASL
        output_path (str): Path to save the output gif
        frames (int): Number of frames to generate
        duration (int): Duration of the gif in seconds
    
    Returns:
        str: Path to the generated gif
    """
    # Create the JSON prompt
    prompt = {
        "input": {
            "text": text_input,
            "parameters": {
                "model": "asl-diffusion-1.0",
                "signer_attributes": {
                    "gender": "neutral",
                    "age_range": "adult",
                    "style": "formal"
                },
                "output_settings": {
                    "format": "gif",
                    "fps": frames // duration,
                    "duration": duration,
                    "background": "neutral",
                    "resolution": "512x512"
                },
                "signing_speed": "medium",
                "include_fingerspelling": True,
                "regional_dialect": "ASL-US-Standard"
            }
        }
    }
    
    print(f"Generating ASL for: '{text_input}'")
    print("Loading model...")
    
    # Load the sign language diffusion model
    try:
        # Try to use the sign language specific model if available
        pipeline = DiffusionPipeline.from_pretrained(
            "Zelda/sign-language-diffusion", 
            torch_dtype=torch.float16
        )
    except:
        # Fall back to a more general text-to-video model
        pipeline = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b", 
            torch_dtype=torch.float16
        )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    print(f"Generating {frames} frames on {device}...")
    
    # Generate the frames
    prompt_text = f"A person signing in American Sign Language: '{text_input}'"
    frames_list = []
    
    # Generate the video frames
    video_frames = pipeline(prompt_text, num_inference_steps=50, num_frames=frames).frames
    
    # Convert to numpy arrays for saving as gif
    frames_list = [np.array(frame) for frame in video_frames]
    
    # Save as gif
    print(f"Saving to {output_path}...")
    imageio.mimsave(output_path, frames_list, fps=frames/duration)
    
    print(f"ASL gif generated and saved to {output_path}")
    return output_path

def display_gif(gif_path):
    """Display the generated gif"""
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as file:
            display(Image(data=file.read()))
    else:
        print(f"File not found: {gif_path}")

# Example usage
if __name__ == "__main__":
    text = input("Enter text to convert to ASL: ")
    gif_path = generate_asl_gif(text)
    display_gif(gif_path)
