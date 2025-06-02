from flask import Flask, render_template, request, redirect, session, url_for, flash
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import cv2
plt.style.use('ggplot')
app = Flask(__name__)
app.secret_key = 'ASL_secret_key'

# Load model
model = tf.keras.models.load_model('SSANL.h5')
class_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# In-memory user store
users = {}

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users:
            flash("Username already exists", "danger")
        else:
            users[uname] = pwd
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users and users[uname] == pwd:
            session['username'] = uname
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            img = Image.open(file).resize((128, 128))
        else:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                flash("Camera capture failed", "danger")
                return redirect(url_for('index'))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((128, 128))

        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = model.predict(img_array)[0]
        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        # Plot chart
        plt.style.use('ggplot')
        fig, ax = plt.subplots()
        ax.barh(class_names, preds, color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_title(f'Prediction: {pred_class} ({confidence:.2f}%)')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return render_template('result.html', pred_class=pred_class, confidence=confidence, chart=image_base64)

    return render_template('predict.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
