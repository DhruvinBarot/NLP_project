import os
import zipfile
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import random

# 1. Create necessary directories
base_dir = "asl_data"
voc_dir = os.path.join(base_dir, "voc")
os.makedirs(base_dir, exist_ok=True)
os.makedirs(voc_dir, exist_ok=True)

# 2. Upload the .voc.zip file to Colab
from google.colab import files
print("Please upload your .voc.zip file...")
uploaded = files.upload()

# Get the name of the uploaded file
uploaded_filename = next(iter(uploaded))
print(f"Uploaded file: {uploaded_filename}")

# 3. Extract the file
print(f"Extracting {uploaded_filename}...")
with zipfile.ZipFile(uploaded_filename, 'r') as zip_ref:
    zip_ref.extractall(base_dir)

# 4. Move all XML and images to the VOC directory
print("Organizing files...")
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.xml') or file.endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(voc_dir, file)
            # Only copy if source and destination are different
            if src_path != dst_path:
                shutil.copy(src_path, dst_path)

# 5. Analyze the dataset
def get_statistics():
    """Get dataset statistics"""
    print("Analyzing dataset...")
    
    # Count images and annotations
    xml_files = [f for f in os.listdir(voc_dir) if f.endswith('.xml')]
    img_files = [f for f in os.listdir(voc_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Total XML annotations: {len(xml_files)}")
    print(f"Total images: {len(img_files)}")
    
    # Count distribution of classes
    class_counts = {}
    for xml_file in tqdm(xml_files, desc="Processing annotations"):
        try:
            tree = ET.parse(os.path.join(voc_dir, xml_file))
            root = tree.getroot()
            for obj in root.findall('.//object'):
                class_name = obj.find('name').text
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
    
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"{cls}: {count}")
    
    # Plot class distribution
    plt.figure(figsize=(15, 8))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=90)
    plt.title('ASL Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'class_distribution.png'))
    print(f"Class distribution chart saved as '{os.path.join(base_dir, 'class_distribution.png')}'")
    
    return class_counts

# 6. Visualize samples
def visualize_samples(num_samples=5):
    """Visualize random samples with bounding boxes"""
    xml_files = [f for f in os.listdir(voc_dir) if f.endswith('.xml')]
    
    if len(xml_files) == 0:
        print("No XML files found!")
        return
    
    # Select random samples
    samples = random.sample(xml_files, min(num_samples, len(xml_files)))
    
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    if len(samples) == 1:
        axes = [axes]
    
    for i, xml_file in enumerate(samples):
        try:
            tree = ET.parse(os.path.join(voc_dir, xml_file))
            root = tree.getroot()
            
            # Get image filename
            img_filename = root.find('.//filename').text
            img_path = os.path.join(voc_dir, img_filename)
            
            # Check if image exists
            if not os.path.exists(img_path):
                print(f"Image {img_filename} not found, skipping")
                continue
            
            # Load and draw image
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            
            # Draw bounding boxes
            for obj in root.findall('.//object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                draw.text((xmin, ymin-15), class_name, fill="red")
            
            # Display in plot
            axes[i].imshow(np.array(img))
            axes[i].set_title(f"ASL: {class_name}")
            axes[i].axis('off')
        except Exception as e:
            print(f"Error processing sample {xml_file}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'sample_visualizations.png'))
    print(f"Sample visualizations saved as '{os.path.join(base_dir, 'sample_visualizations.png')}'")

# 7. Convert to Image Classification format if needed
def convert_to_ic():
    """Convert object detection format to image classification format"""
    try:
        # Check if wai.annotations is installed
        try:
            import importlib
            importlib.import_module('wai.annotations')
        except ImportError:
            print("wai.annotations library not found. Installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wai.annotations==0.7.5"])
            print("Installation complete.")
        
        # Run the conversion command
        import subprocess
        ic_dir = os.path.join(base_dir, "subdir")
        cmd = [
            "wai-annotations", "convert", 
            "from-voc-od", "-i", f"{voc_dir}/*.xml", 
            "od-to-ic", 
            "to-subdir-ic", "-o", ic_dir
        ]
        subprocess.run(cmd, check=True)
        print("Conversion completed successfully!")
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

# 8. Split dataset if needed
def train_test_split(test_size=0.2, val_size=0.1):
    """Split the dataset into train, validation, and test sets"""
    print(f"Splitting dataset with test_size={test_size}, val_size={val_size}")
    
    # Get all XML files
    xml_files = [f for f in os.listdir(voc_dir) if f.endswith('.xml')]
    
    # Create output directories
    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    train_dir = os.path.join(splits_dir, "train")
    val_dir = os.path.join(splits_dir, "val")
    test_dir = os.path.join(splits_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Shuffle files
    random.shuffle(xml_files)
    
    # Calculate split indices
    test_idx = int(len(xml_files) * test_size)
    val_idx = int(len(xml_files) * val_size)
    
    test_files = xml_files[:test_idx]
    val_files = xml_files[test_idx:test_idx+val_idx]
    train_files = xml_files[test_idx+val_idx:]
    
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")
    
    # Copy files to respective directories
    for split_name, file_list, target_dir in [
        ("train", train_files, train_dir),
        ("validation", val_files, val_dir),
        ("test", test_files, test_dir)
    ]:
        for xml_file in tqdm(file_list, desc=f"Copying {split_name} files"):
            try:
                # Copy XML file
                src_xml = os.path.join(voc_dir, xml_file)
                dst_xml = os.path.join(target_dir, xml_file)
                shutil.copy(src_xml, dst_xml)
                
                # Also copy corresponding image
                tree = ET.parse(src_xml)
                root = tree.getroot()
                img_filename = root.find('.//filename').text
                src_img = os.path.join(voc_dir, img_filename)
                
                if os.path.exists(src_img):
                    dst_img = os.path.join(target_dir, img_filename)
                    shutil.copy(src_img, dst_img)
            except Exception as e:
                print(f"Error copying file {xml_file}: {e}")
    
    print("Dataset split completed successfully!")

# Run the data processing
print("Installing required packages...")

# Run analysis
print("\n----- Dataset Analysis -----")
get_statistics()

# Visualize samples
print("\n----- Sample Visualization -----")
visualize_samples(num_samples=5)

# Ask user if they want to convert to IC format
convert_ic = input("Do you want to convert to image classification format? (y/n): ").lower() == 'y'
if convert_ic:
    convert_to_ic()

# Ask user if they want to split the dataset
split_dataset = input("Do you want to split the dataset into train/val/test? (y/n): ").lower() == 'y'
if split_dataset:
    train_test_split()

print("ASL dataset processing completed!")
