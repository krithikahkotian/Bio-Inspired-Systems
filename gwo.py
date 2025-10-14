import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import entropy
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class GreyWolfOptimizer:
    """Grey Wolf Optimizer for image enhancement parameters"""
    
    def __init__(self, n_wolves=10, max_iter=30, dim=5):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        
        # Parameter bounds: [alpha, beta, R, G, B]
        self.lb = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        self.ub = np.array([2.0, 2.0, 1.5, 1.5, 1.5])
        
        # Initialize wolf positions
        self.positions = np.random.uniform(
            self.lb, self.ub, (self.n_wolves, self.dim)
        )
        
        # Alpha, Beta, Delta wolves (best solutions)
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('-inf')
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('-inf')
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('-inf')
        
        self.convergence_curve = []
    
    def optimize(self, fitness_func):
        """Run GWO optimization"""
        print("Starting Grey Wolf Optimization...")
        
        for iteration in range(self.max_iter):
            # Evaluate fitness for all wolves
            for i in range(self.n_wolves):
                fitness = fitness_func(self.positions[i])
                
                # Update Alpha, Beta, Delta
                if fitness > self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                
                elif fitness > self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                
                elif fitness > self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            
            # Linearly decrease 'a' from 2 to 0
            a = 2 - iteration * (2 / self.max_iter)
            
            # Update positions of all wolves
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    # Update using Alpha
                    r1, r2 = np.random.random(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Update using Beta
                    r1, r2 = np.random.random(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Update using Delta
                    r1, r2 = np.random.random(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # Average position
                    self.positions[i, j] = (X1 + X2 + X3) / 3
                    
                    # Boundary check
                    self.positions[i, j] = np.clip(
                        self.positions[i, j], self.lb[j], self.ub[j]
                    )
            
            self.convergence_curve.append(self.alpha_score)
            
            if (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, "
                      f"Best Fitness: {self.alpha_score:.4f}")
        
        print(f"\nOptimization Complete!")
        print(f"Best Parameters: α={self.alpha_pos[0]:.3f}, "
              f"β={self.alpha_pos[1]:.3f}, "
              f"R={self.alpha_pos[2]:.3f}, "
              f"G={self.alpha_pos[3]:.3f}, "
              f"B={self.alpha_pos[4]:.3f}")
        
        return self.alpha_pos, self.alpha_score


class ImageEnhancer:
    """Automatic Image Enhancement System"""
    
    def __init__(self, image_path: str):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.enhanced_image = None
        self.best_params = None
        
        print(f"Image loaded: {self.original_image.shape}")
    
    def apply_enhancement(self, params: np.ndarray, image: np.ndarray = None) -> np.ndarray:
        """Apply enhancement parameters to image"""
        if image is None:
            image = self.original_image
        
        alpha, beta, R, G, B = params
        
        # Convert to float for processing
        enhanced = image.astype(np.float32)
        
        # Adjust brightness and contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=(beta - 1) * 50)
        enhanced = enhanced.astype(np.float32)
        
        # Adjust color balance for each channel
        enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * R, 0, 255)  # R
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * G, 0, 255)  # G
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * B, 0, 255)  # B
        
        return enhanced.astype(np.uint8)
    
    def calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy (information content)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zero probabilities
        return entropy(hist, base=2)
    
    def calculate_edge_intensity(self, image: np.ndarray) -> float:
        """Calculate average edge intensity using Sobel operator"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.mean(edge_magnitude)
    
    def calculate_color_contrast(self, image: np.ndarray) -> float:
        """Calculate color contrast using standard deviation"""
        std_r = np.std(image[:, :, 0])
        std_g = np.std(image[:, :, 1])
        std_b = np.std(image[:, :, 2])
        return (std_r + std_g + std_b) / 3
    
    def fitness_function(self, params: np.ndarray) -> float:
        """
        Fitness function combining entropy, edge intensity, and color contrast
        Higher values indicate better image quality
        """
        try:
            enhanced = self.apply_enhancement(params)
            
            # Calculate quality metrics
            entropy_val = self.calculate_entropy(enhanced)
            edge_intensity = self.calculate_edge_intensity(enhanced)
            color_contrast = self.calculate_color_contrast(enhanced)
            
            # Normalize and combine metrics
            # Weights can be adjusted based on importance
            w1, w2, w3 = 0.4, 0.3, 0.3
            
            fitness = (w1 * entropy_val / 8.0 +  # Normalize entropy (max ~8)
                      w2 * edge_intensity / 100.0 +  # Normalize edge intensity
                      w3 * color_contrast / 100.0)  # Normalize color contrast
            
            return fitness
        
        except Exception as e:
            return float('-inf')
    
    def enhance(self, n_wolves=10, max_iter=30):
        """Perform automatic enhancement using GWO"""
        print("\n" + "="*60)
        print("AUTOMATIC IMAGE ENHANCEMENT USING GREY WOLF OPTIMIZER")
        print("="*60 + "\n")
        
        # Initialize GWO
        gwo = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter, dim=5)
        
        # Run optimization
        self.best_params, best_fitness = gwo.optimize(self.fitness_function)
        
        # Apply best parameters
        self.enhanced_image = self.apply_enhancement(self.best_params)
        
        return self.enhanced_image, self.best_params, gwo.convergence_curve
    
    def calculate_metrics(self) -> dict:
        """Calculate comparison metrics between original and enhanced images"""
        if self.enhanced_image is None:
            raise ValueError("No enhanced image available. Run enhance() first.")
        
        # Convert to grayscale for some metrics
        orig_gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(self.enhanced_image, cv2.COLOR_RGB2GRAY)
        
        metrics = {
            'original_entropy': self.calculate_entropy(self.original_image),
            'enhanced_entropy': self.calculate_entropy(self.enhanced_image),
            'original_edge_intensity': self.calculate_edge_intensity(self.original_image),
            'enhanced_edge_intensity': self.calculate_edge_intensity(self.enhanced_image),
            'psnr': psnr(self.original_image, self.enhanced_image),
            'ssim': ssim(orig_gray, enh_gray),
        }
        
        metrics['entropy_improvement'] = (
            (metrics['enhanced_entropy'] - metrics['original_entropy']) / 
            metrics['original_entropy'] * 100
        )
        
        metrics['edge_improvement'] = (
            (metrics['enhanced_edge_intensity'] - metrics['original_edge_intensity']) / 
            metrics['original_edge_intensity'] * 100
        )
        
        return metrics
    
    def visualize_results(self, convergence_curve: List[float], metrics: dict):
        """Visualize original, enhanced images and metrics"""
        fig = plt.figure(figsize=(18, 10))
        
        # Original Image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(self.original_image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Enhanced Image
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(self.enhanced_image)
        ax2.set_title('Enhanced Image', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Histogram Comparison
        ax3 = plt.subplot(2, 3, 3)
        for i, color in enumerate(['red', 'green', 'blue']):
            hist_orig, _ = np.histogram(self.original_image[:, :, i], bins=256, range=(0, 256))
            hist_enh, _ = np.histogram(self.enhanced_image[:, :, i], bins=256, range=(0, 256))
            ax3.plot(hist_orig, color=color, alpha=0.5, linestyle='--', label=f'Original {color.upper()}')
            ax3.plot(hist_enh, color=color, alpha=0.8, label=f'Enhanced {color.upper()}')
        ax3.set_title('Color Histogram Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Pixel Intensity')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Convergence Curve
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(convergence_curve, linewidth=2, color='#2E86AB')
        ax4.set_title('GWO Convergence Curve', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Fitness Value')
        ax4.grid(True, alpha=0.3)
        
        # Metrics Table
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        metrics_text = f"""
        ENHANCEMENT METRICS
        {'='*40}
        
        Parameters:
        • Brightness (α): {self.best_params[0]:.3f}
        • Contrast (β): {self.best_params[1]:.3f}
        • Red Balance: {self.best_params[2]:.3f}
        • Green Balance: {self.best_params[3]:.3f}
        • Blue Balance: {self.best_params[4]:.3f}
        
        Quality Metrics:
        • Original Entropy: {metrics['original_entropy']:.4f}
        • Enhanced Entropy: {metrics['enhanced_entropy']:.4f}
        • Entropy Improvement: {metrics['entropy_improvement']:.2f}%
        
        • Original Edge Intensity: {metrics['original_edge_intensity']:.2f}
        • Enhanced Edge Intensity: {metrics['enhanced_edge_intensity']:.2f}
        • Edge Improvement: {metrics['edge_improvement']:.2f}%
        
        Comparison Metrics:
        • PSNR: {metrics['psnr']:.2f} dB
        • SSIM: {metrics['ssim']:.4f}
        """
        
        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Edge Detection Comparison
        ax6 = plt.subplot(2, 3, 6)
        orig_gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(self.enhanced_image, cv2.COLOR_RGB2GRAY)
        
        edges_orig = cv2.Canny(orig_gray, 50, 150)
        edges_enh = cv2.Canny(enh_gray, 50, 150)
        
        edges_combined = np.zeros((*edges_orig.shape, 3), dtype=np.uint8)
        edges_combined[:, :, 0] = edges_orig  # Red channel - original
        edges_combined[:, :, 1] = edges_enh   # Green channel - enhanced
        
        ax6.imshow(edges_combined)
        ax6.set_title('Edge Detection\n(Red: Original, Green: Enhanced)', 
                     fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig('enhancement_results.png', dpi=300, bbox_inches='tight')
        print("\nResults saved to 'enhancement_results.png'")
        plt.show()


def main():
    """Main function to run the image enhancement system"""
    
    # Example usage
    image_path = 'input_image.jpg'  # Replace with your image path
    
    try:
        # Create enhancer instance
        enhancer = ImageEnhancer(image_path)
        
        # Perform enhancement
        enhanced_img, best_params, convergence = enhancer.enhance(
            n_wolves=15,  # Number of wolves (search agents)
            max_iter=30   # Number of iterations
        )
        
        # Calculate metrics
        metrics = enhancer.calculate_metrics()
        
        # Visualize results
        enhancer.visualize_results(convergence, metrics)
        
        # Save enhanced image
        cv2.imwrite('enhanced_image.jpg', 
                   cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
        print("\nEnhanced image saved to 'enhanced_image.jpg'")
        
        # Print summary
        print("\n" + "="*60)
        print("ENHANCEMENT SUMMARY")
        print("="*60)
        print(f"Entropy Improvement: {metrics['entropy_improvement']:.2f}%")
        print(f"Edge Intensity Improvement: {metrics['edge_improvement']:.2f}%")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure you have an image file named 'input_image.jpg'")
        print("or modify the image_path variable in the main() function.")


if __name__ == "__main__":
    main()