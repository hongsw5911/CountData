from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import json
import re



# Model configuration
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
prompt_mist = "[INST] <image>\nIdentify the main, prominent, and noticeable objects in the picture, excluding the background, Focus on the primary components. Answer conditions: 1) remove words except nouns 2) answer format is noun,noun,noun... [/INST]"
#"What are Primary, Key, Main objects except background in image?"

dataset = "train"
max_det_noun = 6
batch_size = 16
save_per_iter = 50

# Directory containing the images
image_directory = f"/content/drive/MyDrive/Research/SeoulCountCLIP/coco/{dataset}2017"


# Fetch all image paths from the directory
image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith('.jpg')]


# Path to the results JSON file
output_json_path = f"/content/drive/MyDrive/Research/SeoulCountCLIP/coco/annotations/{dataset}2017_llava_obj_identification.json"

##################################################################################################################################################################


# Initialize the processor and model
processor = LlavaNextProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2"
)
model.to("cuda:0")


# Load existing results if the file exists
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as json_file:
        results_dict = json.load(json_file)
else:
    results_dict = {}


# Dictionary to hold the results
results_dict = {}

# Not object
background_words = [
    "grass", "sand", "wall", "field", "court", "floor", "sky", "wool", "dirt", "street", "mountain", "ground",
    "cloud", "clouds", "mountains", "pavement", "fence", "snow", "ocean", "water", "rain", "beach", "station",
    "platform", "building", "sun", "tracks", "brick", "roof", "bridge", "sidewalk", "dirt road", "wave", "waves",
    "snowy", "path", "buildings", "runway", "mud", "tennis court", "garden", "pathwave", "pathwaves", "rock", "rocks", "zeb",
    "date", "competition", "brick building", "leaves", "shadow", "curb", "smoke", "runway", "sunset", "forest", "ceiling",
    "hill", "soccer field", "skyline", "lights", "light", "brick pavement", "toilet", "bathroom", "hills", "brick road","road",
    "tree","trees","baseball field","stage"
]

# Define the function to process a batch of images
def process_batch(image_paths_batch):
    # Extract filenames
    filenames = [os.path.basename(image_path) for image_path in image_paths_batch]

    # Load images
    images = [Image.open(image_path) for image_path in image_paths_batch]

    # Process the inputs
    inputs = processor(text=[prompt_mist] * len(images), images=images, return_tensors="pt", padding=True).to("cuda:0")

    # Autoregressively complete prompts
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    # Define the function to parse output
    def parse_output(output):
        decoded_output = processor.decode(output, skip_special_tokens=True)
        content_start = decoded_output.split("[/INST]", 1)[1].strip()

        nouns = []
        # parsing nonu of objects
        if "1" in content_start:
    
            # Use regular expressions to find items in the numbered list
            nouns = re.findall(r'\d+\.\s*(\w+)', content_start)
            nouns = [word.strip() for word in nouns]
        elif "features" or "image" in content_start:
            nouns.extend([word.strip() for word in content_start.split(',') if word.strip()])
            nouns = [phrase.split(" ")[-1] for phrase in nouns]
        else:
            nouns.extend([word.strip() for word in content_start.split(',') if word.strip()])
        
        # Filter out empty strings and duplicates
        nouns = list(filter(None, nouns))

        # Remove certain words
        nouns = [word for word in nouns if word not in background_words]
        
        # Lower
        nouns = [word.lower() for word in nouns]
        
        # Remove duplicate words
        nouns = list(set(nouns))[:max_det_noun]

        return nouns
    
    # Use ThreadPoolExecutor to parallelize parsing
    with ThreadPoolExecutor() as executor:
        parsed_results = list(executor.map(parse_output, output))

    # Map filenames to parsed nouns
    for idx, filename in enumerate(filenames):
      if len(parsed_results[idx]):
        results_dict[filename] = parsed_results[idx]
      
# Process all images in batches
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    process_batch(batch_paths)

    if i%save_per_iter == 0:
      # Save results to the JSON file after processing each batch
      if os.path.exists(output_json_path):
          with open(output_json_path, 'r') as json_file:
              existing_results = json.load(json_file)
          existing_results.update(results_dict)
      else:
          existing_results = results_dict
    
      with open(output_json_path, 'w') as json_file:
          json.dump(existing_results, json_file, indent=4)

      print(f"Results saved to {output_json_path} after processing batch {i // batch_size + 1}")
