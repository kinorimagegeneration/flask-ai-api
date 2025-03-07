from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
import torch
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load Stable Diffusion model
print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
print("Model loaded!")

# Define API route for generating images
@app.route("/generate", methods=["POST"])
def generate_image():
    # Get the text prompt from the request
    data = request.json
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Generate image using Stable Diffusion
        print(f"Generating image for prompt: {prompt}")
        image = pipe(prompt).images[0]

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # Return the image as a response
        return send_file(img_byte_arr, mimetype="image/png")

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
