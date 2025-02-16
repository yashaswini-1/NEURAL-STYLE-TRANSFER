import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from PIL import Image

def load_image(image_path, image_size=(256, 256)):
    """Loads and preprocesses an image."""
    image = Image.open(image_path)
    image = image.resize(image_size)
    # Convert to float32 explicitly
    image = np.array(image, dtype=np.float32) / 255.0  
    return np.expand_dims(image, axis=0)

def apply_style_transfer(content_path, style_path):
    """Applies neural style transfer to an image."""
    content_image = load_image(content_path)
    style_image = load_image(style_path)
    
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title('Content Image')
    plt.imshow(content_image[0])
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Style Image')
    plt.imshow(style_image[0])
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Stylized Image')
    plt.imshow(stylized_image[0])
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    content_img_path = "content.jpg"  # Replace with your content image path
    style_img_path = "style.jpg"  # Replace with your style image path
    apply_style_transfer(content_img_path, style_img_path)
