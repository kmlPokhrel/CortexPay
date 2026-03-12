import os
import stripe
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# 1. PRODUCTION OPTIMIZATION (Do this before anything else)
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Limit TensorFlow memory usage for Render's 512MB RAM
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'cortex_secure_prod_99')
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

# 2. FIREBASE SETUP
db = None
try:
    cred_path = os.path.join(os.getcwd(), 'serviceAccountKey.json')
    if not firebase_admin._apps:
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("✅ Firebase Connected")
except Exception as e:
    print(f"🔥 Firebase Error: {e}")

# 3. AI MODEL LOADING (The Version-Proof Way)
model = None
try:
    model_path = os.path.join(os.getcwd(), 'models', 'digit_model.h5')
    # 'compile=False' fixes the 'batch_shape' error caused by TF version differences
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ AI Model Loaded: Version Patch Applied")
except Exception as e:
    print(f"🧠 AI Model Error: {e}")

# --- 4. HELPERS ---
def get_user_data(email):
    if not db: return None
    try:
        user_ref = db.collection('users').document(email).get()
        return user_ref.to_dict() if user_ref.exists else None
    except: return None

# --- 5. ROUTES ---

@app.route('/')
def index():
    if 'user' in session:
        user = get_user_data(session['user'])
        if user and user.get('is_pro'):
            return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/auth')
def auth_page():
    return render_template('login.html', firebase_api_key=os.getenv('FIREBASE_API_KEY'))

@app.route('/firebase-login', methods=['POST'])
def firebase_login():
    if not db: return jsonify({"status": "error"}), 500
    data = request.json
    email, name = data.get('email'), data.get('name')
    db.collection('users').document(email).set({
        'name': name, 'email': email, 'last_login': firestore.SERVER_TIMESTAMP
    }, merge=True)
    session['user'] = email
    return jsonify({"status": "success"})

@app.route('/manual-login', methods=['POST'])
def manual_login():
    if not db: return "Database Error", 500
    email = request.form.get('email')
    db.collection('users').document(email).set({
        'name': email.split('@')[0], 'email': email, 'last_login': firestore.SERVER_TIMESTAMP
    }, merge=True)
    session['user'] = email
    return redirect(url_for('check_access'))

@app.route('/check-access')
def check_access():
    if 'user' not in session: return redirect(url_for('auth_page'))
    user = get_user_data(session['user'])
    if user and user.get('is_pro'): return redirect(url_for('dashboard'))
    return render_template('support.html')

@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user' not in session: return redirect(url_for('auth_page'))
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {'name': 'CortexPay Pro Access'},
                    'unit_amount': 500,
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('payment_success', _external=True),
            cancel_url=url_for('index', _external=True),
        )
        return redirect(checkout_session.url, code=303)
    except Exception as e: return str(e)

@app.route('/payment-success')
def payment_success():
    if 'user' in session and db:
        db.collection('users').document(session['user']).set({'is_pro': True}, merge=True)
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('auth_page'))
    user = get_user_data(session['user'])
    if not user or not user.get('is_pro'): return redirect(url_for('check_access'))
    return render_template('dashboard.html', name=user.get('name', 'User'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session or not model: return jsonify({'error': 'Unauthorized'}), 401
    file = request.files['file']
    img = Image.open(file.stream).convert('L').resize((28, 28))
    if np.mean(np.array(img)) > 127: img = ImageOps.invert(img)
    img_array = (np.array(img) / 255.0).reshape(1, 28, 28)
    
    # Inference
    prediction = int(np.argmax(model.predict(img_array)))
    
    if db:
        db.collection('prediction_logs').add({
            'user': session['user'], 'prediction': prediction, 'timestamp': firestore.SERVER_TIMESTAMP
        })
    return jsonify({'prediction': prediction})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)