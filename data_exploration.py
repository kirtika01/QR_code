import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_dataset(base_path='dataset'):
    """
    Load images from first and second print directories
    Returns two lists of image paths
    """
    first_prints = []
    second_prints = []
    
    # Load first prints
    first_print_path = os.path.join(base_path, 'First Print')
    for img_name in os.listdir(first_print_path):
        if img_name.endswith('.png'):
            first_prints.append(os.path.join(first_print_path, img_name))
            
    # Load second prints
    second_print_path = os.path.join(base_path, 'Second Print')
    for img_name in os.listdir(second_print_path):
        if img_name.endswith('.png'):
            second_prints.append(os.path.join(second_print_path, img_name))
    
    return first_prints, second_prints

def analyze_image_characteristics(image_path):
    """
    Extract basic image characteristics like brightness, contrast, and edge information
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate basic statistics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'edge_density': edge_density
    }

def visualize_differences(first_prints, second_prints):
    """
    Create visualizations comparing first and second prints
    """
    first_stats = []
    second_stats = []
    
    print("Analyzing first prints...")
    for img_path in tqdm(first_prints[:50]):  # Analyze first 50 images
        stats = analyze_image_characteristics(img_path)
        first_stats.append(stats)
    
    print("Analyzing second prints...")
    for img_path in tqdm(second_prints[:50]):  # Analyze first 50 images
        stats = analyze_image_characteristics(img_path)
        second_stats.append(stats)
    
    # Convert to numpy arrays for easier plotting
    first_stats = np.array([[s['brightness'], s['contrast'], s['edge_density']] 
                           for s in first_stats])
    second_stats = np.array([[s['brightness'], s['contrast'], s['edge_density']] 
                            for s in second_stats])
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Brightness distribution
    axes[0].hist(first_stats[:, 0], alpha=0.5, label='First Print', bins=20)
    axes[0].hist(second_stats[:, 0], alpha=0.5, label='Second Print', bins=20)
    axes[0].set_title('Brightness Distribution')
    axes[0].set_xlabel('Brightness')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    
    # Contrast distribution
    axes[1].hist(first_stats[:, 1], alpha=0.5, label='First Print', bins=20)
    axes[1].hist(second_stats[:, 1], alpha=0.5, label='Second Print', bins=20)
    axes[1].set_title('Contrast Distribution')
    axes[1].set_xlabel('Contrast')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    
    # Edge density distribution
    axes[2].hist(first_stats[:, 2], alpha=0.5, label='First Print', bins=20)
    axes[2].hist(second_stats[:, 2], alpha=0.5, label='Second Print', bins=20)
    axes[2].set_title('Edge Density Distribution')
    axes[2].set_xlabel('Edge Density')
    axes[2].set_ylabel('Count')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('data_exploration_results.png')
    plt.close()

def main():
    print("Starting data exploration...")
    
    # Load dataset
    first_prints, second_prints = load_dataset()
    print(f"Found {len(first_prints)} first prints and {len(second_prints)} second prints")
    
    # Visualize differences
    visualize_differences(first_prints, second_prints)
    print("Generated visualization plots in 'data_exploration_results.png'")

if __name__ == "__main__":
    main()