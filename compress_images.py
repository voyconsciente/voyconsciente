import os
from PIL import Image

def compress_image(image_path, output_path, quality=85):
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
        print(f"Imagen comprimida: {output_path}")
    except Exception as e:
        print(f"Error al comprimir imagen: {e}")

# Directorio de imágenes
input_dir = 'static/img'
output_dir = 'static/img/compressed'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.png', '.jpg'))
        compress_image(input_path, output_path)