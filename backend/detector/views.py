from django.shortcuts import render
import joblib
import os
from .text_preprocessing import clean_text

# Load paths for model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sarcasm_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')

# Load once globally
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def index(request):
    prediction = None
    user_input = ''

    if request.method == 'POST':
        user_input = request.POST.get('sentence', '')
        cleaned = clean_text(user_input)

        # DEBUG: See what you're feeding the model
        print("Raw input:", user_input)
        print("Cleaned input:", cleaned)

        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        prediction = "Sarcastic" if pred == 1 else "Not Sarcastic"

    return render(request, 'detector/index.html', {
        'prediction': prediction,
        'input': user_input
    })
