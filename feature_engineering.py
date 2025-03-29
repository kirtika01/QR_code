import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy import stats

class QRFeatureExtractor:
    def __init__(self):
        # Define feature extraction parameters
        self.block_size = 8  # Size of blocks for local analysis
        self.num_blocks = 16  # Number of blocks in each dimension for spatial analysis
        
    def extract_features(self, image_path):
        """
        Extract comprehensive feature set from QR code image
        """
        # Read and preprocess image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # Global features
        features.update(self._extract_global_features(gray))
        
        # Local features
        features.update(self._extract_local_features(gray))
        
        # Texture features
        features.update(self._extract_texture_features(gray))
        
        # Print quality features
        features.update(self._extract_print_quality_features(gray))
        
        return features
    
    def _extract_global_features(self, gray):
        """
        Extract global image statistics
        """
        features = {
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'skewness': stats.skew(gray.flatten()),
            'kurtosis': stats.kurtosis(gray.flatten())
        }
        
        # Global histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features['hist_peak'] = np.argmax(hist)
        features['hist_valley'] = np.argmin(hist)
        
        return features
    
    def _extract_local_features(self, gray):
        """
        Extract features from local image regions
        """
        features = {}
        h, w = gray.shape
        block_h = h // self.num_blocks
        block_w = w // self.num_blocks
        
        local_means = []
        local_stds = []
        
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                local_means.append(np.mean(block))
                local_stds.append(np.std(block))
        
        features['local_mean_std'] = np.std(local_means)
        features['local_std_std'] = np.std(local_stds)
        
        return features
    
    def _extract_texture_features(self, gray):
        """
        Extract texture-based features using GLCM
        """
        features = {}
        
        # Compute gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features['gradient_mean'] = np.mean(magnitude)
        features['gradient_std'] = np.std(magnitude)
        
        # Local Binary Pattern-like features
        binary = (gray > np.mean(gray)).astype(np.uint8)
        features['binary_pattern_ratio'] = np.sum(binary) / binary.size
        
        return features
    
    def _extract_print_quality_features(self, gray):
        """
        Extract features specific to print quality
        """
        features = {}
        
        # Edge sharpness
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Noise estimation
        noise = cv2.fastNlMeansDenoising(gray)
        features['noise_level'] = np.mean(np.abs(gray - noise))
        
        # Contrast measures
        features['local_contrast'] = np.mean(np.abs(gray - cv2.GaussianBlur(gray, (5,5), 0)))
        
        return features

def prepare_dataset(base_path='dataset'):
    """
    Process all images and create feature vectors
    """
    extractor = QRFeatureExtractor()
    features = []
    labels = []
    
    # Process first prints
    print("Processing first prints...")
    first_print_path = os.path.join(base_path, 'First Print')
    for img_name in tqdm(os.listdir(first_print_path)):
        if img_name.endswith('.png'):
            img_path = os.path.join(first_print_path, img_name)
            feature_dict = extractor.extract_features(img_path)
            features.append(list(feature_dict.values()))
            labels.append(0)  # 0 for first print
    
    # Process second prints
    print("Processing second prints...")
    second_print_path = os.path.join(base_path, 'Second Print')
    for img_name in tqdm(os.listdir(second_print_path)):
        if img_name.endswith('.png'):
            img_path = os.path.join(second_print_path, img_name)
            feature_dict = extractor.extract_features(img_path)
            features.append(list(feature_dict.values()))
            labels.append(1)  # 1 for second print
    
    return np.array(features), np.array(labels)

def main():
    print("Starting feature extraction...")
    features, labels = prepare_dataset()
    
    # Save features and labels
    np.save('features.npy', features)
    np.save('labels.npy', labels)
    print(f"Extracted features from {len(features)} images")
    print(f"Feature vector dimension: {features.shape[1]}")

if __name__ == "__main__":
    main()