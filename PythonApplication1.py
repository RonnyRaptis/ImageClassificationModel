# Ronald Raptis

import os
import time
import cv2
import yt_dlp
import requests
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# retrieve api keys/credientials from os
training_key = os.getenv('VISION_TRAINING_KEY')
endpoint = os.getenv('VISION_TRAINING_ENDPOINT')
prediction_resource_id = os.getenv('VISION_PREDICTION_RESOURCE_ID')
prediction_key = os.getenv('VISION_PREDICTION_KEY')
prediction_endpoint = os.getenv('VISION_PREDICTION_ENDPOINT')

# verify training/prediction clients
trainer = CustomVisionTrainingClient(endpoint, ApiKeyCredentials(in_headers={"Training-key": training_key}))
predictor = CustomVisionPredictionClient(prediction_endpoint, ApiKeyCredentials(in_headers={"Prediction-key": prediction_key}))

# existing customvison.ai project info
existing_project_name = "TrafficModel"
publish_iteration_name = "TrafficModel"

# find existing project
all_projects = trainer.get_projects()
project = next((p for p in all_projects if p.name == existing_project_name), None)

if project is None:
    raise Exception(f"No existing project named '{existing_project_name}' found. Please check your Azure Custom Vision portal.")

print(f"Found existing project: {project.name} (ID: {project.id})")

# option A: youtube live stream integration
def get_youtube_stream_url(youtube_url):
    """Extracts a direct stream URL from a YouTube link using yt_dlp."""
    ydl_opts = {"format": "best", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]

def capture_frame_from_stream(stream_url, max_retries=5, initial_delay=2):
    """Captures a single frame from a live stream URL using OpenCV."""
    retries = 0
    while retries < max_retries:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("Error: Could not open video stream. Retrying...")
            retries += 1
            time.sleep(initial_delay * (2 ** (retries - 1)))  # exponential backoff
            continue

        # allow a brief moment for the stream to stabilize
        time.sleep(2)
        ret, frame = cap.read()
        cap.release()

        if ret:
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                return buffer.tobytes()
        print(f"Error: Could not capture frame. Retrying {retries + 1}/{max_retries}")
        retries += 1
        time.sleep(initial_delay * (2 ** (retries - 1)))  # exponential backoff

    raise Exception("Failed to capture a frame from the live stream after multiple attempts.")

# option B: NYC traffic camera integration
def fetch_nyc_traffic_image(image_url):
    """Downloads the latest traffic snapshot from an NYC traffic camera."""
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            return response.content  # returns image bytes
        else:
            print(f"Error: Unable to fetch image (Status Code: {response.status_code})")
            return None
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

# unified classification function
def classify_traffic_image(source_type, source_value):
    """
    Classifies an image from either:
    - a YouTube live stream (source_type='youtube', source_value=YouTube URL)
    - an NYC traffic camera (source_type='nyc_camera', source_value=Camera Image URL)
    """
    try:
        if source_type == 'youtube':
            stream_url = get_youtube_stream_url(source_value)
            image_data = capture_frame_from_stream(stream_url)
        elif source_type == 'nyc_camera':
            image_data = fetch_nyc_traffic_image(source_value)
        else:
            raise ValueError("Invalid source type. Use 'youtube' or 'nyc_camera'.")

        if image_data:
            results = predictor.classify_image(project.id, publish_iteration_name, image_data)
            print("Prediction Results:")
            for prediction in results.predictions:
                print(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")
        else:
            print("Failed to retrieve image data.")
    except Exception as e:
        print(f"Error processing image: {e}")

#run classification for either YT or NYC cameras
source_choice = "youtube"  # change to 'youtube' or 'nyc_camera'
youtube_url = "https://www.youtube.com/live/9En2186vo5g?si=a2lvMSh983uxCj69"                                          #insert valid YT live stream url here
nyc_camera_url = "https://webcams.nyctmc.org/api/cameras/f2b94b32-fc41-42b6-935d-8330374ca05a/image?t=1743453086914"  #insert valid NYC camera url here

if source_choice == "youtube":
    classify_traffic_image("youtube", youtube_url)
elif source_choice == "nyc_camera":
    classify_traffic_image("nyc_camera", nyc_camera_url)
else:
    print("Invalid source choice. Please use 'youtube' or 'nyc_camera'.")
