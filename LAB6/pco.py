import numpy as np
import cv2
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import os

class ParallelCellularAlgorithm:
    """
    Parallel Cellular Algorithm for Image Enhancement
    Optimizes brightness and contrast parameters
    """
    
    def __init__(self, image_path, grid_size=10, max_iterations=200):
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"Loaded image: {self.original_image.shape}")
        
        # Convert to grayscale for processing
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # PCA Parameters
        self.grid_size = grid_size
        self.total_cells = grid_size * grid_size
        self.max_iterations = max_iterations
        
        # Parameter ranges
        self.B_MIN, self.B_MAX = -50, 50  # Brightness range
        self.C_MIN, self.C_MAX = 0.5, 2.0  # Contrast range
        
        # Fitness weights
        self.w1 = 0.4  # Entropy weight
        self.w2 = 0.4  # Edge density weight
        self.w3 = 0.2  # MSE penalty weight
        
        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size), dtype=object)
        self.initialize_grid()
        
        # Tracking
        self.best_B = 0
        self.best_C = 1.0
        self.best_fitness = -float('inf')
        self.fitness_history = []
        self.best_params_history = []
        
    def initialize_grid(self):
        """Initialize each cell with random B and C values"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                B = np.random.uniform(self.B_MIN, self.B_MAX)
                C = np.random.uniform(self.C_MIN, self.C_MAX)
                self.grid[i, j] = {'B': B, 'C': C, 'fitness': 0}
    
    def apply_enhancement(self, image, B, C):
        """
        Apply brightness and contrast adjustment
        Formula: output = C * (input - 128) + 128 + B
        """
        # Convert to float for precision
        enhanced = image.astype(np.float32)
        
        # Apply contrast around midpoint (128) then add brightness
        enhanced = C * (enhanced - 128.0) + 128.0 + B
        
        # Clip to valid range [0, 255]
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def calculate_entropy(self, image):
        """Calculate Shannon entropy of the image"""
        histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        histogram = histogram / histogram.sum()  # Normalize
        
        # Remove zeros to avoid log(0)
        histogram = histogram[histogram > 0]
        
        # Shannon entropy
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy
    
    def calculate_edge_density(self, image):
        """Calculate edge density using Sobel operator"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Compute gradients using Sobel
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude of gradient
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge density (normalized)
        edge_density = np.mean(gradient_magnitude) / 255.0
        return edge_density
    
    def calculate_mse(self, original, enhanced):
        """Calculate Mean Squared Error (normalized)"""
        mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
        return mse / (255.0 * 255.0)  # Normalize to [0, 1]
    
    def fitness_function(self, B, C):
        """
        Fitness function to evaluate image quality
        F(B,C) = w1*Entropy + w2*EdgeDensity - w3*MSE
        """
        # Apply enhancement to grayscale image
        enhanced = self.apply_enhancement(self.gray_image, B, C)
        
        # Calculate metrics
        entropy = self.calculate_entropy(enhanced)
        edge_density = self.calculate_edge_density(enhanced)
        mse = self.calculate_mse(self.gray_image, enhanced)
        
        # Compute fitness (higher is better)
        fitness = self.w1 * entropy + self.w2 * edge_density - self.w3 * mse
        
        return fitness
    
    def get_moore_neighbors(self, i, j):
        """Get Moore neighborhood (8 neighbors) with toroidal wrap-around"""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                # Wrap around (toroidal topology)
                ni = ni % self.grid_size
                nj = nj % self.grid_size
                neighbors.append((ni, nj))
        return neighbors
    
    def update_cell(self, i, j):
        """Update cell based on best neighbor using diffusion rule"""
        current_cell = self.grid[i, j]
        neighbors = self.get_moore_neighbors(i, j)
        
        # Find best neighbor
        best_neighbor_fitness = current_cell['fitness']
        best_neighbor = current_cell
        
        for ni, nj in neighbors:
            neighbor = self.grid[ni, nj]
            if neighbor['fitness'] > best_neighbor_fitness:
                best_neighbor_fitness = neighbor['fitness']
                best_neighbor = neighbor
        
        # Update using diffusion rule (move toward best neighbor)
        alpha = 0.3  # Learning rate
        new_B = current_cell['B'] + alpha * (best_neighbor['B'] - current_cell['B'])
        new_C = current_cell['C'] + alpha * (best_neighbor['C'] - current_cell['C'])
        
        # Add small random exploration
        new_B += np.random.normal(0, 2)
        new_C += np.random.normal(0, 0.05)
        
        # Clip to valid ranges
        new_B = np.clip(new_B, self.B_MIN, self.B_MAX)
        new_C = np.clip(new_C, self.C_MIN, self.C_MAX)
        
        return {'B': new_B, 'C': new_C, 'fitness': 0}
    
    def evaluate_grid(self):
        """Evaluate fitness for all cells in parallel"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                cell['fitness'] = self.fitness_function(cell['B'], cell['C'])
                
                # Track global best
                if cell['fitness'] > self.best_fitness:
                    self.best_fitness = cell['fitness']
                    self.best_B = cell['B']
                    self.best_C = cell['C']
    
    def run(self, verbose=True):
        """Run the PCA optimization"""
        if verbose:
            print("\n" + "="*60)
            print("Starting Parallel Cellular Algorithm")
            print("="*60)
            print(f"Grid Size: {self.grid_size}x{self.grid_size} ({self.total_cells} cells)")
            print(f"Max Iterations: {self.max_iterations}")
            print(f"Brightness Range: [{self.B_MIN}, {self.B_MAX}]")
            print(f"Contrast Range: [{self.C_MIN}, {self.C_MAX}]")
            print("-" * 60)
        
        import time
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Evaluate all cells
            self.evaluate_grid()
            
            # Store history
            self.fitness_history.append(self.best_fitness)
            self.best_params_history.append((self.best_B, self.best_C))
            
            # Update all cells (parallel update - synchronous)
            new_grid = np.zeros((self.grid_size, self.grid_size), dtype=object)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    new_grid[i, j] = self.update_cell(i, j)
            
            self.grid = new_grid
            
            # Print progress
            if verbose and (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1:3d}/{self.max_iterations} | "
                      f"Fitness: {self.best_fitness:7.4f} | "
                      f"B: {self.best_B:6.2f} | C: {self.best_C:5.3f}")
        
        end_time = time.time()
        
        if verbose:
            print("-" * 60)
            print(f"Optimization Complete!")
            print(f"Time: {end_time - start_time:.2f} seconds")
            print(f"Best Fitness: {self.best_fitness:.4f}")
            print(f"Optimal B*: {self.best_B:.2f}")
            print(f"Optimal C*: {self.best_C:.3f}")
            print("="*60 + "\n")
        
        return self.best_B, self.best_C
    
    def get_enhanced_image(self, B=None, C=None):
        """
        Get enhanced image using specified or optimal parameters
        Applies enhancement to full color image
        """
        if B is None:
            B = self.best_B
        if C is None:
            C = self.best_C
        
        # Apply to each channel of the color image
        enhanced_color = self.original_image.copy()
        
        for channel in range(3):
            enhanced_color[:, :, channel] = self.apply_enhancement(
                self.original_image[:, :, channel], B, C
            )
        
        return enhanced_color
    
    def visualize_results(self, save_path=None):
        """Visualize original, enhanced images and convergence"""
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Enhanced image
        ax2 = fig.add_subplot(gs[0:2, 1])
        enhanced = self.get_enhanced_image()
        ax2.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Enhanced Image\nB* = {self.best_B:.2f}, C* = {self.best_C:.3f}', 
                     fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        # Histograms comparison
        ax3 = fig.add_subplot(gs[0:2, 2])
        ax3.hist(self.gray_image.flatten(), bins=256, range=(0, 256), 
                alpha=0.5, color='blue', label='Original', density=True)
        enhanced_gray = self.apply_enhancement(self.gray_image, self.best_B, self.best_C)
        ax3.hist(enhanced_gray.flatten(), bins=256, range=(0, 256), 
                alpha=0.5, color='red', label='Enhanced', density=True)
        ax3.set_xlabel('Pixel Intensity', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Histogram Comparison', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Fitness convergence
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(self.fitness_history, linewidth=2, color='#2E86AB')
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Best Fitness', fontsize=11)
        ax4.set_title('Fitness Convergence', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Parameter evolution
        ax5 = fig.add_subplot(gs[2, 1])
        B_history = [p[0] for p in self.best_params_history]
        C_history = [p[1] for p in self.best_params_history]
        
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(B_history, linewidth=2, color='#A23B72', label='Brightness')
        ax5.set_xlabel('Iteration', fontsize=11)
        ax5.set_ylabel('Brightness (B)', fontsize=11, color='#A23B72')
        ax5.tick_params(axis='y', labelcolor='#A23B72')
        
        line2 = ax5_twin.plot(C_history, linewidth=2, color='#F18F01', label='Contrast')
        ax5_twin.set_ylabel('Contrast (C)', fontsize=11, color='#F18F01')
        ax5_twin.tick_params(axis='y', labelcolor='#F18F01')
        
        ax5.set_title('Parameter Evolution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Metrics comparison
        ax6 = fig.add_subplot(gs[2, 2])
        
        # Calculate metrics for both images
        orig_entropy = self.calculate_entropy(self.gray_image)
        orig_edge = self.calculate_edge_density(self.gray_image)
        
        enh_entropy = self.calculate_entropy(enhanced_gray)
        enh_edge = self.calculate_edge_density(enhanced_gray)
        
        metrics = ['Entropy', 'Edge\nDensity']
        orig_vals = [orig_entropy/8, orig_edge]  # Normalize entropy for display
        enh_vals = [enh_entropy/8, enh_edge]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax6.bar(x - width/2, orig_vals, width, label='Original', color='#4A90E2')
        ax6.bar(x + width/2, enh_vals, width, label='Enhanced', color='#E94B3C')
        
        ax6.set_ylabel('Normalized Value', fontsize=11)
        ax6.set_title('Quality Metrics', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics, fontsize=10)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Image Enhancement using Parallel Cellular Algorithm', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to: {save_path}")
        
        plt.show()
    
    def save_enhanced_image(self, output_path):
        """Save the enhanced image"""
        enhanced = self.get_enhanced_image()
        cv2.imwrite(output_path, enhanced)
        print(f"Enhanced image saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # IMPORTANT: Replace 'your_image.jpg' with your actual image path
    image_path = 'download.jpg'  # <-- CHANGE THIS
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file '{image_path}' not found!")
        print("Please provide a valid image path.")
        print("\nCreating a sample image for demonstration...")
        
        # Create a more realistic sample image (dark, low contrast)
        sample = np.ones((400, 600, 3), dtype=np.uint8) * 100
        # Add some structure
        sample[50:150, 50:250] = 140
        sample[200:350, 300:550] = 120
        sample = cv2.GaussianBlur(sample, (15, 15), 0)
        
        image_path = 'sample_low_contrast.jpg'
        cv2.imwrite(image_path, sample)
        print(f"Sample image created: {image_path}")
    
    try:
        # Initialize PCA
        pca = ParallelCellularAlgorithm(
            image_path=image_path,
            grid_size=10,
            max_iterations=200
        )
        
        # Run optimization
        optimal_B, optimal_C = pca.run(verbose=True)
        
        # Visualize results
        pca.visualize_results(save_path='enhancement_results.png')
        
        # Save enhanced image
        pca.save_enhanced_image('enhanced_image.jpg')
        
        print("\nAll outputs saved successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your image path and try again.")