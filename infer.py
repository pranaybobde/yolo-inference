from ultralytics import YOLO
import os
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["POST"])
def test():
    data = request.get_json()

    if "dir" not in data:
        return jsonify({"error": "Missing 'dir' parameter in the request"}), 400
    if "model_path" not in data:
        return jsonify({"error": "Missing 'model path' parameter in the request"}), 400
    
    dir = data["dir"]
    model_path = data["model_path"]

    model = YOLO(model_path)

    # dir = "D:/Personal/yolo_v8/yolov8_models_infer/sample_1"
    # model = YOLO("D:/Personal/yolo_v8/yolov8_models_infer/yolov8n.pt")

    img_files = [f for f in os.listdir(dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
    images = []
    for i, f in enumerate(img_files): 
        img_path = os.path.join(dir, f)
        img = cv2.imread(img_path)
        images.append(img)
    
    result = model.predict(source=images, save=True)
    print(result)
    return "tested !!" ,201


if __name__ == "__main__":
    app.run(debug=True)