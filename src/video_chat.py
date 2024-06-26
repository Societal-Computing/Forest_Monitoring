import cv2
import os
import base64
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import csv

API_KEY = "sk-proj-VvCSf7uhVqzOJFOIMcx7T3BlbkFJP5u2x4zp8ApYKXaBEtMm"

def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frames = []
    while success:
        cv2.imwrite(f"frame{count}.jpg", image)  # Save frame as JPEG file
        frames.append(f"frame{count}.jpg")
        success, image = vidcap.read()
        count += 1
    return frames

def encode_image(image_path):
    with Image.open(image_path) as img:
        with BytesIO() as buffer:
            img.convert('RGB').save(buffer, 'JPEG')
            jpeg_bytes = buffer.getvalue()
            base64_str = base64.b64encode(jpeg_bytes).decode()
            return base64_str

def get_gpt_description(base64_image, api_key, prompt_text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt_text}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def get_text(response):
    text = response['choices'][0]['message']['content']
    return text

if __name__ == '__main__':
    video_path = '/home/idisc02/Saarland_Forest_monitoring_research/Reforestation_Monitoring/Forest_Detection/polygon_15/video/output_video.avi'
    frames = extract_frames(video_path)
    
    with open('/home/idisc02/Saarland_Forest_monitoring_research/Reforestation_Monitoring/Forest_Detection/polygon_15/video/video_analysis.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'description'])
        
        for frame in frames:
            base64_image = encode_image(frame)
            gpt_response = get_gpt_description(base64_image, API_KEY, "Describe the changes in tree cover in each  video frame   compared to the previous one, estimate the tree cover percentage in each frame  also please my video has only four frames")
            text = get_text(gpt_response)
            writer.writerow([frame, text])
            os.remove(frame)  # Optional: remove frame file after processing