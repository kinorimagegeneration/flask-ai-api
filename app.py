from flask import Flask, request, jsonify, send_file
import torch
from dalle_mini import DalleBart, DalleBartProcessor
from PIL import Image
import io

app = Flask(__name__)

# Load AI Model
model_name = "dalle-mini/dalle-mini"
processor = DalleBartProcessor.from_pretrained(model_name)
model = DalleBart.from_pretrained(model_name, torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.get_json()
    text_prompt = data.get("prompt", "A fantasy landscape with floating islands")
    
    # Process input text
    inputs = processor(text_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate image
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Convert to PIL Image
    image = Image.fromarray(outputs[0].cpu().numpy(), 'RGB')

    # Save the image to a buffer
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
