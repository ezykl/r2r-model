from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model/r2r_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Define supported image formats
ALLOWED_EXTENSIONS = {"jpeg", "jpg", "png", "gif"}

# Define class names
class_names = [
    "Action Camera", "Adjustable Wrench", "Air Compressor", "Air Mover", "Angle Grinder",
    "Audio Mixer", "Axe", "Ball", "Belt Sander", "Bench Grinder",
    "Beverages", "Bike & E-Bike", "Book", "Boom Microphone", "Bounce House",
    "Brad Nailer", "Butane & LPG", "Camera Tripod", "Camping Cot", "Camping Tent",
    "Canned & Packaged Food", "Car Battery Charger", "Car Jack", "Car Polisher",
    "Caulking Gun", "Chainsaw", "Circular Saw", "Clamp Meter", "Claw Hammer",
    "Combination Square", "Combination Wrench Set", "Concrete Mixer", "Concrete Saw",
    "Cordless Drill", "Cosmetics Products", "Crimping Tool", "Crowbar", "DJ Controller",
    "DSLR camera", "Drain Auger", "Drone", "Drywall Screw Gun", "Drywall Trowel",
    "Dust Extractor", "Earth Auger", "Electric Hoist", "Electric Planer",
    "Electric ScrewDriver", "Engine Hoist", "Everyday Clothing", "Explosive",
    "Extension Cord Reel", "Extension Ladder", "Finishing Nailer", "Firearms",
    "Flathead Screwdriver Set", "Floor Polisher", "Floor Scraper", "Folding Camping Chair",
    "Folding Chairs", "Folding Tables", "Fresh & Frozen Food", "Gambling Items",
    "Garden Rake", "Garden Tiller", "Gimbal Stabilizer", "Hacksaw", "Hammer Drill",
    "Hand Saw", "Hand Truck", "Hand-to-Hand Combat Weapon", "Hard Hat", "Hazardous Waste",
    "Headphone", "Heat Gun", "Heavy-Duty Fan", "Hedge Trimmer", "Hoe", "Houses & Apartments",
    "Identity documents", "Impact Wrench", "Infrared Thermometer", "Jack Stands", "Jackhammer",
    "Jewelry & Personal Accessories", "Jigsaw", "Ladder", "Laptop", "Laser Level",
    "Lavalier Microphone", "Lawn Mower", "Leaf Blower", "Lighter", "Livestock", "Loppers",
    "Medical Textiles Product", "Metal Detector", "Metal Shears", "Microphone Stand", "Money",
    "Monitor", "Motorbike & Scooter", "Mouthguards", "Multimeter", "Needle Nose Pliers",
    "Office Space", "Oral Care Products", "PEX Crimping Tool", "PVC Cutter", "Pet",
    "Phillips Screwdriver Set", "Pickaxe", "Pickup Truck", "Pipe Threader", "Pipe Wrench",
    "Pop-up Canopy Tent", "Portable Camping Stove", "Portable Cooler", "Portable Generator",
    "Portable Green Screen", "Post Hole Digger", "Power Trowel", "Prescription Drugs & Medicine",
    "Pressure Washer", "Projector", "Projector Screen", "Prosthetic Equipments", "Putty Knife",
    "Rebar Cutter", "Rental Car", "Ring Light", "Router Tool", "Rubber Mallet", "Scaffold Tower",
    "Scissor", "Served Food", "Shoes & Footwear", "Shop Vacuum", "Shovel", "Sledgehammer",
    "Sleeping Bag", "Slip Joint Pliers", "Socket Wrench Set", "Socks & Hosiery", "Sound System",
    "Spirit Level", "Steam Cleaner", "Storage Unit & Locker", "String Trimmer", "Stroller",
    "Studio Softbox Kit", "Syringes & Needles", "Tape Measure", "Thermal Camera", "Tin Snips",
    "Tire Inflator", "Tool Box", "Tool Chest", "Torque Wrench", "Torx Screwdriver Set",
    "Towels & Bathrobes", "Undergarments", "Utility Knife", "Vacuum Cleaner", "Vise Grip",
    "Voltage Tester", "Wallpaper Steamer", "Water Pump", "Welding Clamps", "Welding Helmet",
    "Welding Machine", "Welding Table", "Wet Tile Saw", "Wire Strippers", "Wire Tracer",
    "Wireless Microphone Kit", "Wireless Speaker", "Wood Carving Kit", "Wood Hand Planer",
    "Work Light"
]

# Define prohibited classes
prohibited_classes = {
    "Beverages", "Bike & E-Bike", "Butane & LPG", "Canned & Packaged Food",
    "Cosmetics Products", "Everyday Clothing", "Explosive", "Firearms",
    "Fresh & Frozen Food", "Gambling Items", "Hand-to-Hand Combat Weapon",
    "Hazardous Waste", "Houses & Apartments", "Identity documents",
    "Jewelry & Personal Accessories", "Lighter", "Livestock",
    "Medical Textiles Product", "Money", "Motorbike & Scooter", "Mouthguards",
    "Office Space", "Oral Care Products", "Pet", "Pickup Truck",
    "Prescription Drugs & Medicine", "Prosthetic Equipments", "Rental Car",
    "Served Food", "Shoes & Footwear", "Socks & Hosiery",
    "Storage Unit & Locker", "Syringes & Needles", "Towels & Bathrobes",
    "Undergarments"
}

# Check if file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Test route to check if API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running successfully!"})
    
# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Get image from request
        image = request.files["image"]

        # Check allowed formats
        if not allowed_file(image.filename):
            return jsonify({"error": "Unsupported file format. Use JPEG, JPG, PNG, or GIF."}), 400

        # Open image and convert if necessary
        try:
            img = Image.open(image)
        except UnidentifiedImageError:
            return jsonify({"error": "Invalid image file."}), 400

        # Convert GIF to RGB if necessary
        if img.format == "GIF":
            img = img.convert("RGB")

        # Resize and preprocess image
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)[0]
        top_3_indices = np.argsort(prediction)[-3:][::-1]  # Top 3 predictions
        highest_confidence = max(prediction)

        # If confidence ≥ 70%, return the highest prediction
        if highest_confidence >= 0.7:
            top_index = top_3_indices[0]
            predicted_item = class_names[top_index]
            category = "Prohibited" if predicted_item in prohibited_classes else "Accepted"
            return jsonify([{
                "Predicted Item": predicted_item,
                "Category": category,
                "Confidence": f"{highest_confidence * 100:.2f}%"
            }])

        # If confidence < 70% but ≥ 20%, return top 3 predictions
        results = []
        if highest_confidence >= 0.2:
            for i in top_3_indices:
                predicted_item = class_names[i]
                category = "Prohibited" if predicted_item in prohibited_classes else "Accepted"
                confidence = f"{prediction[i] * 100:.2f}%"

                results.append({
                    "Predicted Item": predicted_item,
                    "Category": category,
                    "Confidence": confidence
                })
            return jsonify(results)

        # If confidence < 20%, classify as 'Unknown'
        return jsonify([{
            "Predicted Item": "Unknown",
            "Category": "N/A",
            "Confidence": f"{highest_confidence * 100:.2f}% (Low Confidence)"
        }])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Flask server 
# Activate environment and install dependencies using the command: .\env\Scripts\activate
# Start the Flask server using the command: python app.py
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
