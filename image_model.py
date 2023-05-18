import numpy as np
from network import Network, sgn
from PIL import Image
import cv2

# this script is certainly not optimized, for the record.

def relabel(mat: np.ndarray) -> np.ndarray:
    new = np.empty((50, 50))
    for i in range(50):
        for j in range(50):
            new[i, j] = sgn(mat[i, j] - 127.5) # = 1 if closer to 255, = -1 if closer to 0.
    return new

def restore(mat: np.ndarray) -> np.ndarray:
    new = np.empty((50, 50), dtype=np.uint8)
    for i in range(50):
        for j in range(50):
            new[i, j] = (mat[i, j] + 1) * 127.5 # = 127.5 * 2 or 127.5 * 0
    return new

pi = relabel(cv2.imread('training_images/pi.png', 0))
sigma = relabel(cv2.imread('training_images/sigma.png', 0))
theta = relabel(cv2.imread('training_images/theta.png', 0))

image_model = Network(patterns=[pi, sigma, theta])

pi_distorted = relabel(cv2.imread('samples/inputs/pi_distorted.png', 0))
sigma_distorted = relabel(cv2.imread('samples/inputs/sigma_distorted.png', 0))
theta_distorted = relabel(cv2.imread('samples/inputs/theta_distorted.png', 0))

# random_input = np.random.randint(2, size=(50, 50), dtype=np.int16)
# random_input[test_input == 0] = -1

pi_restored = Image.fromarray(restore(image_model.compute(pi_distorted)))
sigma_restored = Image.fromarray(restore(image_model.compute(sigma_distorted)))
theta_restored = Image.fromarray(restore(image_model.compute(theta_distorted)))


pi_restored.save('samples/outputs/pi_restored.png')
sigma_restored.save('samples/outputs/sigma_restored.png')
theta_restored.save('samples/outputs/theta_restored.png')