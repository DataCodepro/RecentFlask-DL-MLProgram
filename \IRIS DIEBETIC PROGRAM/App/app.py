from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
import torch
from torchvision import transforms
from PIL import Image
from models import DualIrisNet  # Ensure this points to your actual model

app = Flask(__name__)
app.secret_key = "iris_secert_key"

# Dummy user store (replace with a real database in production)
users = {
    "admin": {"password": "password"}  # Replace with secure hash in real use
}

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# Load model
model = DualIrisNet()
model.load_state_dict(torch.load("dual_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route("/")
def index():
    return render_template("base.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users:
            flash("Username already exists.")
        else:
            users[username] = {"password": password}
            flash("Registration successful. Please log in.")
            return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = users.get(username)

        if user and user["password"] == password:
            login_user(User(username))
            flash("Logged in successfully.")
            return redirect(url_for("predict"))
        else:
            flash("Invalid username or password.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for("index"))

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction = None

    if request.method == "POST":
        if "iris_image" not in request.files:
            flash("No image uploaded.")
            return redirect(request.url)

        file = request.files["iris_image"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        try:
            image = Image.open(file).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                bio_out, _ = model(image_tensor)
                pred_class = torch.argmax(bio_out, dim=1).item()
                prediction = "Diabetic" if pred_class == 1 else "Control"

        except Exception as e:
            flash(f"Prediction failed: {e}")
            return redirect(request.url)

    return render_template("predict.html", username=current_user.username, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
