import os
import csv
import base64
import textwrap
import requests
import logging
from PIL import Image
from io import BytesIO
import re

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = "sk-proj-VvCSf7uhVqzOJFOIMcx7T3BlbkFJP5u2x4zp8ApYKXaBEtMm"

def get_config(api_key, base64_images, prompt_text):
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
                    "content": prompt_text
                },
                *[
                    {
                        "role": "system",
                        "content": f"data:image/jpeg;base64,{image}"
                    } for image in base64_images
                ]
            ],
            "max_tokens": 500
        }
    }
    return config

def encode_image(geotiff_path):
    with Image.open(geotiff_path) as img:
        with BytesIO() as buffer:
            img.convert('RGB').save(buffer, 'JPEG')
            jpeg_bytes = buffer.getvalue()
            base64_str = base64.b64encode(jpeg_bytes).decode()
            return base64_str

def get_gpt_description(image_groups, api_key, prompt_text):
    responses = []
    for image_paths in image_groups:
        base64_images = [encode_image(image_path) for image_path in image_paths]
        config = get_config(api_key, base64_images, prompt_text)
        headers = config["headers"]
        payload = config["payload"]
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                responses.append(response.json())
            else:
                logging.error(f"Failed API request with status code {response.status_code}: {response.text}")
                responses.append({'error': 'Failed to fetch description'})
        except Exception as e:
            logging.error(f"Exception during API request: {str(e)}")
            responses.append({'error': 'Exception during API request'})
    return responses

def get_text(response):
    if 'error' in response:
        return response['error']
    text = response['choices'][0]['message']['content']
    width = 60
    paragraph = textwrap.fill(text, width=width)
    return paragraph

def list_image_files_with_year(folder_path):
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.tif', '.jpg', '.jpeg', '.png'))]
    grouped_files = {}
    for file in files:
        year_match = re.search(r'(\d{4})\d{4}_', file)
        if year_match:
            year = year_match.group(1)
            if year in ['2019', '2021', '2022', '2024']:
                if year not in grouped_files:
                    grouped_files[year] = []
                grouped_files[year].append(file)
    return [(year, grouped_files[year]) for year in sorted(grouped_files)]

if __name__ == '__main__':
    folder_path = '/home/idisc02/Saarland_Forest_monitoring_research/Reforestation_Monitoring/Forest_Detection/polygon_15'
    image_groups = dict(list_image_files_with_year(folder_path))
    year_pairs = [('2019', '2021'), ('2021', '2022'), ('2022', '2024')]

    with open('Forest_changes.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_group', 'comparison', 'response'])

        for year1, year2 in year_pairs:
            if year1 in image_groups and year2 in image_groups:
                prompt_text = f" given {year1} satellite image and {year2} satellite image help me to briefly  Describe the changes in tree cover in the satellite images from {year1} to {year2}."
                images = image_groups[year1] + image_groups[year2]
                gpt_responses = get_gpt_description([images], API_KEY, prompt_text)
                for response in gpt_responses:
                    text = get_text(response)
                    writer.writerow(["; ".join(os.path.basename(img) for img in images), f"{year1}-{year2}", text])