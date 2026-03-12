import os
import stripe
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# 1. INITIALIZATION
load_dotenv()
app = Flask(__name__)
# Secret key for session encryption
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'prod_secret_88')
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

# --- 2. FIREBASE SETUP (With safety check) ---
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"🔥 Firebase Error: {e}. Check if serviceAccountKey.json is missing!")

# --- 3. AI MODEL LOADING ---
try:
    model = tf.keras.models.load_model('models/digit_model.h5')
except Exception as e:
    print(f"🧠 AI Model Error: {e}. Check if 'models/digit_model.h5' exists!")

# --- 4. LOGIC HELPERS ---
def get_user_data(email):
    try:
        user_ref = db.collection('users').document(email).get()
        return user_ref.to_dict() if user_ref.exists else None
    except:
        return None

# --- 5. ROUTES ---

@app.route('/')
def index():
    # If already logged in and pro, skip the landing page
    if 'user' in session:
        user = get_user_data(session['user'])
        if user and user.get('is_pro'):
            return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/auth')
def auth_page():
    # Pass API key for Google Login
    return render_template('login.html', firebase_api_key=os.getenv('FIREBASE_API_KEY'))

@app.route('/firebase-login', methods=['POST'])
def firebase_login():
    data = request.json
    email, name = data.get('email'), data.get('name')
    # Sync Google User to Firestore
    db.collection('users').document(email).set({
        'name': name, 
        'email': email, 
        'last_login': firestore.SERVER_TIMESTAMP
    }, merge=True)
    session['user'] = email
    return jsonify({"status": "success"})

@app.route('/manual-login', methods=['POST'])
def manual_login():
    email = request.form.get('email')
    # Manual login simulation for the demo
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
    # If they paid, go to AI. If not, go to Payment page.
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
                    'product_data': {'name': 'Pro AI Access'}, 
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
    if 'user' in session:
        # Finalize Pro status in Cloud DB
        db.collection('users').document(session['user']).set({'is_pro': True}, merge=True)
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('auth_page'))
    user = get_user_data(session['user'])
    # Strict Guard: Only Pros allowed here
    if not user or not user.get('is_pro'): 
        return redirect(url_for('check_access'))
    return render_template('dashboard.html', name=user.get('name', 'User'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session: return jsonify({'error': 'Unauthorized'}), 401
    
    file = request.files['file']
    img = Image.open(file.stream).convert('L').resize((28, 28))
    
    # Pre-processing
    if np.mean(np.array(img)) > 127: 
        img = ImageOps.invert(img)
    img_array = (np.array(img) / 255.0).reshape(1, 28, 28)
    
    # AI Thinking
    res = int(np.argmax(model.predict(img_array)))
    
    # Cloud Logging
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

if __name__ == '__main__':
    # FORCING PORT 5000 
    app.run(debug=True, port=5000)