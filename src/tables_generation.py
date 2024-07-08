import os
import csv
import base64
import textwrap
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

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
    with Image.open(geotiff_path) as img:
        with BytesIO() as buffer:
            img.convert('RGB').save(buffer, 'JPEG')
            jpeg_bytes = buffer.getvalue()
            base64_str = base64.b64encode(jpeg_bytes).decode()
            return base64_str

def get_gpt_description(image_paths, api_key, prompt_text):
    base64_images = [encode_image(image_path) for image_path in image_paths]
    config = get_config(api_key, base64_images, prompt_text)
    headers = config["headers"]
    payload = config["payload"]
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def get_text(response):
    text = response['choices'][0]['message']['content']
    width = 60
    paragraph = textwrap.fill(text, width=width)
    return paragraph

def list_image_files(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.tif', '.jpg', '.jpeg', '.png'))]
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
    folder_path = '/home/idisc02/Saarland_Forest_monitoring_research/Reforestation_Monitoring/Forest_Detection/class1'
    image_files = list_image_files(folder_path)
   
    with open('forest_new.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
     
        writer.writerow(['image_name', 'response'])
        
        for image_file in image_files:
            gpt_response = get_gpt_description([image_file], API_KEY, " What percentage of the satellite image cover trees,please give only the percentage in the table")
            text = get_text(gpt_response)
            display_image_with_description(image_file, text)
            
          
            writer.writerow([os.path.basename(image_file), text])