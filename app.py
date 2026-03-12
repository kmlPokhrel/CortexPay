import os
import stripe
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# 1. CLOUD OPTIMIZATION
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress heavy AI logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Stability for small RAM

app = Flask(__name__)
# Use environment variable for secret key, fallback only for local
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key_for_dev')
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

# 2. FIREBASE SETUP (Production Hardened)
db = None
try:
    # Render looks for this file in the root directory
    cred_path = os.path.join(os.getcwd(), 'serviceAccountKey.json')
    if not firebase_admin._apps:
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("✅ Firebase Connected: Production Mode")
        else:
            print("❌ Error: serviceAccountKey.json not found in root!")
except Exception as e:
    print(f"🔥 Firebase Initialization Failed: {e}")

# 3. AI MODEL LOADING
model = None
try:
    model_path = os.path.join(os.getcwd(), 'models', 'digit_model.h5')
    model = tf.keras.models.load_model(model_path)
    print("✅ AI Model Loaded Successfully")
except Exception as e:
    print(f"🧠 AI Model Error: {e}")

# --- 4. LOGIC HELPERS ---
def get_user_data(email):
    if not db: return None
    try:
        user_ref = db.collection('users').document(email).get()
        return user_ref.to_dict() if user_ref.exists else None
    except:
        return None

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
    if not db: return jsonify({"status": "error", "message": "Database not initialized"}), 500
    data = request.json
    email, name = data.get('email'), data.get('name')
    db.collection('users').document(email).set({
        'name': name, 
        'email': email, 
        'last_login': firestore.SERVER_TIMESTAMP
    }, merge=True)
    session['user'] = email
    return jsonify({"status": "success"})

@app.route('/manual-login', methods=['POST'])
def manual_login():
    if not db: return "Database connection error", 500
    email = request.form.get('email')
    db.collection('users').document(email).set({
        'name': email.split('@')[0], 
        'email': email, 
        'last_login': firestore.SERVER_TIMESTAMP
    }, merge=True)
    session['user'] = email
    return redirect(url_for('check_access'))

@app.route('/check-access')
def check_access():
    if 'user' not in session: return redirect(url_for('auth_page'))
    user = get_user_data(session['user'])
    if user and user.get('is_pro'):
        return redirect(url_for('dashboard'))
    return render_template('support.html')

@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user' not in session: return redirect(url_for('auth_page'))
    try:
        session_stripe = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd', 
                    'product_data': {'name': 'CortexPay Pro Access'}, 
                    'unit_amount': 500
                }, 
                'quantity': 1
            }],
            mode='payment',
            success_url=url_for('payment_success', _external=True),
            cancel_url=url_for('index', _external=True),
        )
        return redirect(session_stripe.url, code=303)
    except Exception as e:
        return f"Stripe Error: {e}"

@app.route('/payment-success')
def payment_success():
    if 'user' in session and db:
        db.collection('users').document(session['user']).set({'is_pro': True}, merge=True)
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('auth_page'))
    user = get_user_data(session['user'])
    if not user or not user.get('is_pro'): 
        return redirect(url_for('check_access'))
    return render_template('dashboard.html', name=user.get('name', 'User'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session: return jsonify({'error': 'Unauthorized'}), 401
    
    file = request.files['file']
    img = Image.open(file.stream).convert('L').resize((28, 28))
    
    if np.mean(np.array(img)) > 127: 
        img = ImageOps.invert(img)
    img_array = (np.array(img) / 255.0).reshape(1, 28, 28)
    
    res = int(np.argmax(model.predict(img_array)))
    
    if db:
        db.collection('prediction_logs').add({
            'user': session['user'], 
            'prediction': res, 
            'timestamp': firestore.SERVER_TIMESTAMP
        })
    return jsonify({'prediction': res})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# --- 6. DYNAMIC PORT BINDING (The "Fix") ---
if __name__ == '__main__':
    # Use the port assigned by the cloud provider, default to 5000 for local
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)