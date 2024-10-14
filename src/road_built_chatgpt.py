import os
import csv
import base64
import textwrap
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import re
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

API_KEY = "sk-proj-VvCSf7uhVqzOJFOIMcx7T3BlbkFJP5u2x4zp8ApYKXaBEtMm"

def get_config(api_key, base64_image, prompt_text):
    config = {
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        "payload": {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt_text}"
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
    }
    content = []
    for i in range(len(base64_image)):
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image[i]}"
                }
            }
        )
    config["payload"]["messages"][0]["content"].extend(content)
    return config

def encode_image(geotiff_path):
    try:
        with Image.open(geotiff_path) as img:
            with BytesIO() as buffer:
                img.convert('RGB').save(buffer, 'JPEG')
                jpeg_bytes = buffer.getvalue()
                base64_str = base64.b64encode(jpeg_bytes).decode()
                return base64_str
    except Exception as e:
        print(f"Error encoding image {geotiff_path}: {str(e)}")
        return None

def get_gpt_description(image_paths, api_key, prompt_text):
    base64_images = [encode_image(image_path) for image_path in image_paths]
    
    # Filtering out any images that couldn't be encoded
    base64_images = [img for img in base64_images if img is not None]
    if not base64_images:
        print("No valid images to process for GPT.")
        return None

    config = get_config(api_key, base64_images, prompt_text)
    headers = config["headers"]
    payload = config["payload"]
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Logging the response for debugging
    print(f"API response: {response.json()}")
    
    return response.json()

def get_text(response):
    if 'choices' in response:
        text = response['choices'][0]['message']['content']
        width = 60
        paragraph = textwrap.fill(text, width=width)
        return paragraph
    elif 'error' in response:
        print(f"Error from GPT API: {response['error']['message']}")
        return None
    else:
        print("Unexpected response format:", response)
        return None

def list_image_files_by_folder(base_folder):
    image_files = []
    if os.path.exists(base_folder):
        image_files = [os.path.join(base_folder, file) for file in os.listdir(base_folder) if file.endswith(('.tif', '.jpg', '.jpeg', '.png'))]
    return image_files

def mask_black_boundary(image_path):
  
    img = cv2.imread(image_path)
    
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    
 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    return masked_img

def process_images_and_generate_csv(base_folder, api_key):
    image_files = list_image_files_by_folder(base_folder)
    

    data = {}
    
    
    site_pattern = re.compile(r"(reforest_site_\d+)")
    
    
    displayed_images_count = 0
    max_images_to_display = 5  # Limiting the display to the first 5 images
    
    # Processing each image
    for image_file in image_files:
        # Extracting site name using regex (matches "reforest_site_" followed by digits)
        match = site_pattern.search(os.path.basename(image_file))
        if match:
            site_name = match.group(1)
        else:
            site_name = "Unknown_site"
        
       
        masked_img = mask_black_boundary(image_file)
        
        # Saving the masked image temporarily in memory and pass it for GPT processing
        temp_img_path = '/tmp/temp_masked_image.jpg' 
        cv2.imwrite(temp_img_path, masked_img)  
        
        # The GPT response for the masked image,and the prompts
        gpt_response = get_gpt_description([temp_img_path], api_key, "Are there 1. road networks, 2. built areas? Please respond with 'Yes' if they exist and 'No' if they do not.")
        
        if gpt_response is None:
            print(f"Skipping {image_file} due to error in GPT response.")
            continue
        
        description = get_text(gpt_response)
        
        if description is None:
            print(f"Skipping {image_file} due to error in description parsing.")
            continue

      
        print(f"GPT response for {image_file}: {description}")
        
        # Using the regex fxn to extract "Yes" or "No" for road networks and built areas
        pattern = re.compile(r'1\. Road networks: (Yes|No)\s+2\. Built areas: (Yes|No)')
        match = pattern.search(description)
        
        if match:
            
            road_exists = match.group(1)
            built_exists = match.group(2)
        else:
            print(f"Warning: Unexpected format in GPT response for {image_file}. Defaulting to 'No'.")
            road_exists, built_exists = "No", "No"
        
     
        if displayed_images_count < max_images_to_display:
            display_image_with_description(temp_img_path, f"Road Networks: {road_exists}\nBuilt Areas: {built_exists}")
            displayed_images_count += 1
  
        data[site_name] = {'road_network': road_exists, 'built_areas': built_exists}
    
    # Saving Output data to CSV file
    csv_path = '/Users/angela/Documents/NICFI_No_built_area/chatgpt_built.csv'
    write_to_csv(csv_path, data)

def write_to_csv(filepath, data):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
  
        writer.writerow(['Site', 'Road Networks', 'Built Areas'])
    
        for site, values in data.items():
            writer.writerow([site, values['road_network'], values['built_areas']])

def display_image_with_description(image_path, description):
    img = Image.open(image_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')  
    plt.subplot(1, 2, 2)
    plt.text(0, 0.5, description, wrap=True)
    plt.axis('off') 
    plt.show()

if __name__ == '__main__':
    base_folder = '/Users/angela/Documents/NICFI_No_built_area'
    process_images_and_generate_csv(base_folder, API_KEY)
