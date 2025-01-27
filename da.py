import time
import os
import telebot
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow


# Initialize Roboflow API
rf = Roboflow(api_key="zUmGkrDO82Ep4CuEul5G")
project = rf.workspace("colorant").project("colorant-detection-v2maf")
version = project.version(2)
dataset = version.download("multiclass")

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="zUmGkrDO82Ep4CuEul5G"
)

# Initialize Telegram bot
API_TOKEN = '7550160938:AAGAdvfaw-q5OnaHRYyI-6ZAXAxJ1ij94HU'
bot = telebot.TeleBot(API_TOKEN)

# Function to process the image with Roboflow and filter by confidence


def process_image(image_path, confidence_threshold=0.5):
    try:
        # Perform inference
        result = CLIENT.infer(
            image_path, model_id="colorant-detection-v2maf/2")

        # Log the result for debugging
        print("Roboflow response:", result)

        # Extract predictions
        predictions = result.get("predictions", {})
        if not predictions:
            return ["No detections found"]

        # Filter predictions by confidence threshold
        labels = []
        for label, details in predictions.items():
            confidence = details.get('confidence', 0)
            if confidence >= confidence_threshold:
                labels.append(label)

        # If no labels above the threshold, return a default message
        if not labels:
            return ["No confident detections found"]

        return labels
    except Exception as e:
        print(f"Error in process_image: {e}")
        return ["Error: Unable to process the image"]

# Telegram bot handler for images


@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        # Download the image from the user
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        image_path = f"{message.photo[-1].file_id}.jpg"

        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        # Process the image
        labels = process_image(image_path)

        # Send the detected labels back to the user
        response = "Detected: " + ", ".join(labels)
        bot.send_message(message.chat.id, response)

        # Clean up saved image
        if os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        print(f"Error handling image: {e}")
        bot.send_message(
            message.chat.id, "Sorry, an error occurred while processing your image.")


# Resilient bot polling with retry logic
while True:
    try:
        print("Bot is running...")
        bot.polling()
    except Exception as e:
        print(f"Bot polling error: {e}")
        time.sleep(5)  # Wait before restarting polling
