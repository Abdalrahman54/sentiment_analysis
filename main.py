from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
from nltk.stem import PorterStemmer
import tensorflow as tf
from transformers import BertTokenizer, TFDistilBertForSequenceClassification

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Porter Stemmer
stemmer = PorterStemmer()

# Preprocessing function
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+', '', text)
    return ' '.join(stemmer.stem(word) for word in text.split())

# Load the tokenizer and model (TensorFlow version)
def get_model():
    try:
        tokenizer = BertTokenizer.from_pretrained(r'E:\wepbsite\app\tokenaizer_sentment')
        model = TFDistilBertForSequenceClassification.from_pretrained(r'E:\wepbsite\app\pretrained_sentment')

        print("Model and Tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

# Load the tokenizer and model globally (handle potential errors)
tokenizer, model = get_model()
if tokenizer is None or model is None:
    print("Failed to load model and tokenizer. Exiting.")
    exit() # Exit if loading fail

# Class mapping
class_mapping = {0: "positive", 1: "neutral", 2: "negative", 3: "irrelevant"}

# Prediction endpoint
@app.post("/predict")
async def predict(request: Request):
    if tokenizer is None or model is None: # Check if model loaded correctly
        return {"Error": "Model not loaded."}
    data = await request.json()
    if 'text' in data:
        user_input = data['text']
        preprocessed_text = [preprocess_text(user_input)]

        encoded_inputs = tokenizer(
            preprocessed_text,
            padding=True,
            truncation=True,
            return_tensors="tf"
        )

        if 'token_type_ids' in encoded_inputs:
            del encoded_inputs['token_type_ids']

        try:  # Add a try-except block for prediction errors
            predictions = model(encoded_inputs)
            logits = predictions.logits
            probs = tf.nn.softmax(logits, axis=-1)
            predicted_class_index = tf.argmax(probs, axis=-1).numpy()[0]
            predicted_class = class_mapping[predicted_class_index]

            response = {
                "Received Text": user_input,
                "Preprocessed Text": preprocessed_text[0],
                "Predicted Class": predicted_class,
                "Class Probabilities": {
                    label: float(prob) for label, prob in zip(class_mapping.values(), probs.numpy()[0])
                }
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            response = {"Error": "Prediction failed."}
        return response

    else:
        return {"Error": "No 'text' field found in request body."}

# Run the application
if __name__ == "__main__":  
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
