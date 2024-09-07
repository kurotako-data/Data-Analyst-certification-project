from PIL import Image
import os

# Chemin vers le dossier images
image_folder = './images/'

# Fonction pour convertir les images en PNG
def convert_images_to_png(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('jpeg', 'jpg', 'bmp')):
                image_path = os.path.join(root, file)
                img = Image.open(image_path)
                new_image_path = os.path.splitext(image_path)[0] + ".png"
                img.save(new_image_path, 'png')
                print(f"Converted {image_path} to {new_image_path}")

# Exécute la fonction de conversion
convert_images_to_png(image_folder)

