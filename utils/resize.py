import os
from PIL import Image

def batch_scale_images(input_dir, output_dir, scale_factor=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                img = Image.open(input_path)
                original_width, original_height = img.size
                
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                new_size = (new_width, new_height)
                
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS) 
                
                img_resized.save(output_path)
                print(f"  ✅ Succeed: {filename} ({original_width}x{original_height} -> {new_width}x{new_height})")
                
            except Exception as e:
                print(f"  ❌ Failed to process {filename}: {e}")

INPUT_FOLDER = '/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/SA-1B/sa_000005' 
OUTPUT_FOLDER = '/gemini/space/yifq/zhaozy/ousiqu/attn/datasets/SA-1B/resize/sa_000005'
SCALE_FACTOR = 0.65 

batch_scale_images(INPUT_FOLDER, OUTPUT_FOLDER, SCALE_FACTOR)
print("Done!")