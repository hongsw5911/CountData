from transformers import AutoProcessor, CLIPSegForImageSegmentation, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
import torch
import numpy as np
import os 
import json 
import shutil  
import matplotlib.pyplot as plt

# seg hyper params
threshold_1 = 0.03 
threshold_2 = 0.03
threshold_3 = 0.1
hideCategory = False
cateColor = np.array([255, 255, 255]) # black RGB as numpy array
dataset = "train"

# detection hyper params
bbox_threshold = 0.15
text_confidence = 0.30
iou_threshold = 0.98
det_steps = 2

# file, dir path
json_file = "/content/drive/MyDrive/Research/SeoulCountCLIP/coco/annotations/train2017_llava_obj_identification.json" # Object identification json file
image_directory = f"/content/drive/MyDrive/Research/SeoulCountCLIP/coco/{dataset}2017" # Directory containing the images
seg_output_dir = "/content/drive/MyDrive/Research/SeoulCountCLIP/segResult" # Save the segmentation results

# Device set
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the processor and model
processor_seg = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model_seg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
processor_det = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model_det = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

# Load the JSON data
with open(json_file, 'r') as file:
    data = json.load(file)

# Extract keys from each object
existing_filenames = list(data.keys())

# Fetch all image paths from the directory
image_paths_names = [(os.path.join(image_directory, filename), filename) for filename in os.listdir(image_directory) if filename.endswith('.jpg') and existing_filenames]

# Function to reset the output directory
def reset_output_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Reset the output directory
reset_output_dir(seg_output_dir)

# modified iou definition
def calculate_modified_iou(box_i, box_j):
    x1, y1, x2, y2 = box_i
    x1_p, y1_p, x2_p, y2_p = box_j

    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(x1, x1_p)
    yA = max(y1, y1_p)
    xB = min(x2, x2_p)
    yB = min(y2, y2_p)

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the bounding boxes
    box_i_Area = (x2 - x1 ) * (y2 - y1 )
    box_j_Area = (x2_p - x1_p ) * (y2_p - y1_p)

    # Use the area of the smaller bounding box as the denominator
    smaller_area = min(box_i_Area, box_j_Area)

    # Compute the modified IoU
    modified_iou = interArea / float(smaller_area)

    return modified_iou, box_i_Area, box_j_Area

# Function to apply segmentation mask to bounding boxes
def apply_segmentation_mask(image, segmentation_mask, boxes):

    # Create a copy of the image to modify
    image_arr = np.array(image)

    # Iterate over each box and apply the segmentation mask
    for box in boxes:
        
        x1, y1, x2, y2 = map(int, box)
        #overlap_mask = segmentation_mask[y1:y2, x1:x2]
        #ask == 1] = cateColor  # Change the overlap area to black  
        image_arr[y1:y2, x1:x2,:] = cateColor  # Change the overlap area to black  

    return Image.fromarray(image_arr)


def background_segmentation_1(processor_seg, model_seg, object_list, image, device, threshold_1, hideCategory, cateColor, seg_output_dir, file_name):

      # Prepare the inputs
    inputs = processor_seg(text=object_list, images=[image] * len(object_list), padding=True, return_tensors="pt").to(device)

    # Generate the outputs
    with torch.no_grad():
        outputs = model_seg(**inputs)

    logits = outputs.logits
  
    # Process the logits to create segmentation masks
    logits = logits.sigmoid().cpu().numpy()
    masks = (logits > threshold_1).astype(np.uint8)  # Binarize the masks

    # 1st, Iterate over each text and corresponding mask
    objectwise_image_list = []
    for i, (one_object, objectwise_mask) in enumerate(zip(object_list, masks)):
        # Resize mask to match the original image size
        objectwise_mask = Image.fromarray(objectwise_mask * 255)  # Convert binary mask to a PIL image
        objectwise_mask = objectwise_mask.resize(image.size, resample=Image.NEAREST)  # Resize the mask to match the original image size
        objectwise_mask = np.array(objectwise_mask) / 255  # Convert mask back to a binary numpy array

        image_arr = np.array(image)
        if hideCategory:
            image_arr[objectwise_mask == 1] = cateColor  # Broadcast operation
        else:
            image_arr[objectwise_mask == 0] = cateColor  # Broadcast operation

        one_obj_semantic = Image.fromarray(image_arr)
        objectwise_image_list.append((one_object, one_obj_semantic))
        one_obj_semantic.save(f"{seg_output_dir}/{file_name}_{one_object.replace(' ', '_')}_preSeg_1.jpg")

    return objectwise_image_list

def background_segmentation_2(processor_seg, model_seg, objectwise_image_list, image, device, threshold_2, hideCategory, cateColor, seg_output_dir, file_name ):
    

    # 2nd, Iterate over each text and corresponding mask
    detection_candidate = []
    for one_object, objectwisely_masked_image in objectwise_image_list:
        # Prepare the inputs
        one_obj_input = processor_seg(text=[one_object], images=[objectwisely_masked_image], padding=True, return_tensors="pt").to(device)

        # Generate the outputs
        with torch.no_grad():
            one_obj_output = model_seg(**one_obj_input)

        logit = one_obj_output.logits.squeeze()
        # Process the logits to create segmentation masks
        logit = logit.sigmoid().cpu().numpy()
        objectwise_mask = (logit > threshold_2).astype(np.uint8)  # Binarize the masks

        # Resize mask to match the original image size
        objectwise_mask = Image.fromarray(objectwise_mask * 255)  # Convert binary mask to a PIL image
        objectwise_mask = objectwise_mask.resize(image.size, resample=Image.NEAREST)  # Resize the mask to match the original image size
        objectwise_mask = np.array(objectwise_mask) / 255  # Convert mask back to a binary numpy array

        one_obj_image_arr = np.array(objectwisely_masked_image)

        if hideCategory:
            one_obj_image_arr[objectwise_mask == 1] = cateColor  # Broadcast operation
        else:
            one_obj_image_arr[objectwise_mask == 0] = cateColor  # Broadcast operation

        one_obj_semantic = Image.fromarray(one_obj_image_arr)
        detection_candidate.append((one_object,one_obj_semantic))
        one_obj_semantic.save(f"{seg_output_dir}/{file_name}_{one_object.replace(' ', '_')}_preSeg_2.jpg")
        
    return detection_candidate


def detection_and_object_segmentation(processor_seg, model_seg, processor_det, model_det, device, detection_candidate, image, threshold_3, bbox_threshold, text_confidence, seg_output_dir, file_name, step):
  
    # 3rd, perform object detection
    next_det = []
    for one_object, objectwisely_masked_image in detection_candidate:


        # Prepare the inputs
        one_obj_input = processor_seg(text=[one_object], images=[objectwisely_masked_image], padding=True, return_tensors="pt").to(device)

        # Generate the outputs
        with torch.no_grad():
            one_obj_output = model_seg(**one_obj_input)

        logit = one_obj_output.logits.squeeze()
        # Process the logits to create segmentation masks
        logit = logit.sigmoid().cpu().numpy()
        objectwise_mask = (logit > threshold_3).astype(np.uint8)  # Binarize the masks

        # Resize mask to match the original image size
        objectwise_mask = Image.fromarray(objectwise_mask * 255)  # Convert binary mask to a PIL image
        objectwise_mask = objectwise_mask.resize(image.size, resample=Image.NEAREST)  # Resize the mask to match the original image size
        objectwise_mask = np.array(objectwise_mask) / 255  # Convert mask back to a binary numpy array

    
        
        inputs = processor_det(images=objectwisely_masked_image, text=one_object + "." , return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_det(**inputs)

        results = processor_det.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=bbox_threshold,
            text_threshold=text_confidence,
            target_sizes=[image.size[::-1]]
        )

        # Filter results based on the conditions
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
            if score >= text_confidence and label != '':
                
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)

        # Remove nested bounding boxes based on modified IoU
        if len(filtered_boxes) ==1:

          final_boxes = filtered_boxes
          final_scores = filtered_scores
          final_labels = filtered_labels

        else:

          idx_set = set(range(len(filtered_boxes)))
          indices_to_remove = set()


          for i in range(len(filtered_boxes)):
              for j in range(len(filtered_boxes)):
                  if i  < j:            
                      modified_iou, area_i, area_j = calculate_modified_iou(filtered_boxes[i], filtered_boxes[j])
                      if modified_iou > iou_threshold:      
                        if area_i > area_j:
                          indices_to_remove.add(i)
                        else:
                          indices_to_remove.add(j)
              
          # Remove the identified indices
          idx_set = idx_set - indices_to_remove
          final_boxes = [filtered_boxes[idx] for idx in idx_set]
          final_scores = [filtered_scores[idx] for idx in idx_set]
          final_labels = [filtered_labels[idx] for idx in idx_set]

        final_image = apply_segmentation_mask(objectwisely_masked_image, objectwise_mask, final_boxes)
        final_image.save(f"{seg_output_dir}/{file_name}_{one_object.replace(' ', '_')}_objSeg_{step}.jpg")
        next_det.append((one_object, final_image))

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(objectwisely_masked_image)
        font = ImageFont.load_default()
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            draw.rectangle(box.tolist(), outline="red", width=1)
            draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red", font=font)

        # Save the image with bounding boxes to local path
        objectwisely_masked_image.save(f"{seg_output_dir}/{file_name}_{one_object.replace(' ', '_')}_objFet_{step}.jpg")

    return next_det

# main code
for img_seq,(file_path, file_name) in enumerate(image_paths_names):

    object_list = data[file_name]
    image = Image.open(file_path)

    objectwise_image_list = background_segmentation_1(processor_seg, model_seg, object_list, image, device, threshold_1, hideCategory, cateColor, seg_output_dir, file_name)
    detection_candidate = background_segmentation_2(processor_seg, model_seg, objectwise_image_list, image, device, threshold_2, hideCategory, cateColor, seg_output_dir, file_name)
    #for step in range(det_steps):
    next_detection_candidate = detection_and_object_segmentation(processor_seg, model_seg, processor_det, model_det, device, detection_candidate, image, threshold_3, bbox_threshold, text_confidence, seg_output_dir, file_name, 1)
    detection_and_object_segmentation(processor_seg, model_seg, processor_det, model_det, device, next_detection_candidate, image, threshold_3, bbox_threshold, text_confidence, seg_output_dir, file_name, 2)

    


    """
    # 3rd, perform object detection
    for one_object, objectwisely_masked_image in detection_candidate:
        one_object = one_object + "." 
        
        inputs = processor_det(images=objectwisely_masked_image, text=one_object, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_det(**inputs)

        results = processor_det.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=bbox_threshold,
            text_threshold=text_confidence,
            target_sizes=[image.size[::-1]]
        )

       
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(objectwisely_masked_image)
        font = ImageFont.load_default()

        for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
            draw.rectangle(box.tolist(), outline="red", width=1)
            draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red", font=font)

        # Save the image with bounding boxes to local path
        objectwisely_masked_image.save(f"{seg_output_dir}/{img_seq}_{one_object.replace(' ', '_')}_3.jpg")
    """


    