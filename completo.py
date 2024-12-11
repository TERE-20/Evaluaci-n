from flask import Flask, render_template
from flask_sock import Sock
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io

# Configuracion de Flask
app = Flask(__name__)
sock = Sock(app)

# Cargar el modelo YOLO
model = YOLO('/home/ubuntu/plagas/plagas/best.pt')

@app.route('/')
def home():
    return render_template('plag.html')

@sock.route('/detect')
def detect(ws):
    while True:
        message = ws.receive()
        if message is None:
            break

        # Si se recibe una imagen base64 desde el cliente
        if isinstance(message, str):
            # Decodificar la imagen base64
            img_data = base64.b64decode(message)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img_array = np.frombuffer(message, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Procesar la imagen con YOLO
        results = model(img)
        for result in results:
            annotated_img = result.plot()

        # Codificar la imagen para enviarla de vuelta
        _, buffer = cv2.imencode('.jpg', annotated_img)
        ws.send(buffer.tobytes())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
