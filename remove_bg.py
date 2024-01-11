from rembg import remove 
from PIL import Image 
  
input_path = "./finger_videos/testing/1.png"
  
output_path = "./finger_videos/testing/1-remove-bg.png"
  
input = Image.open(input_path) 
  
output = remove(input) 
output.save(output_path) 