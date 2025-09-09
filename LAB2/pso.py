import cv2
import numpy as np
import os
from skimage.measure import shannon_entropy
from skimage.filters import sobel


# ---------------- Fitness Function ----------------
def fitness_function(image, alpha=0.7, beta=0.3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(gray)
    edges = sobel(gray)
    edge_strength = np.mean(edges)
    return alpha * entropy + beta * edge_strength


# ---------------- Image Enhancement ----------------
def enhance_image(img, contrast, brightness):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


# ---------------- Particle Swarm Optimization ----------------
def pso_optimize(image, num_particles=15, max_iter=30, alpha=0.7, beta=0.3):
    contrast_range = (0.5, 3.0)
    brightness_range = (-50, 50)

    particles = np.random.rand(num_particles, 2)
    particles[:, 0] = contrast_range[0] + particles[:, 0] * (contrast_range[1] - contrast_range[0])
    particles[:, 1] = brightness_range[0] + particles[:, 1] * (brightness_range[1] - brightness_range[0])

    velocities = np.random.uniform(-1, 1, (num_particles, 2))

    personal_best_positions = particles.copy()
    personal_best_scores = np.array([
        fitness_function(enhance_image(image, c, b), alpha, beta)
        for c, b in particles
    ])

    global_best_index = np.argmax(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_score = personal_best_scores[global_best_index]

    w, c1, c2 = 0.7, 1.5, 1.5

    for iteration in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - particles[i])
                + c2 * r2 * (global_best_position - particles[i])
            )
            particles[i] += velocities[i]

            particles[i, 0] = np.clip(particles[i, 0], *contrast_range)
            particles[i, 1] = np.clip(particles[i, 1], *brightness_range)

            enhanced = enhance_image(image, particles[i, 0], particles[i, 1])
            score = fitness_function(enhanced, alpha, beta)

            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i].copy()

        best_idx = np.argmax(personal_best_scores)
        if personal_best_scores[best_idx] > global_best_score:
            global_best_score = personal_best_scores[best_idx]
            global_best_position = personal_best_positions[best_idx].copy()

        print(f"Iteration {iteration+1}/{max_iter}, Best Score: {global_best_score:.4f}")

    return global_best_position, global_best_score


# ---------------- Interactive Run ----------------
def main():
    # Ask user for input image
    input_path = input("Enter input image file name (with extension, e.g., input.jpg): ").strip()

    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # Auto-generate output file name
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_enhanced{ext}"

    # Load image
    image = cv2.imread(input_path)

    # Run PSO
    best_params, best_score = pso_optimize(image)
    best_contrast, best_brightness = best_params
    print(f"Optimal Contrast: {best_contrast:.3f}, Optimal Brightness: {best_brightness:.3f}")

    # Apply enhancement
    enhanced = enhance_image(image, best_contrast, best_brightness)

    # Save output
    cv2.imwrite(output_path, enhanced)
    print(f"Enhanced image saved as {output_path}")


if __name__ == "__main__":
    main()
