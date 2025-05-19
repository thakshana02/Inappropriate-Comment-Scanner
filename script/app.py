
from flask import Flask, render_template, request, jsonify
import os
import re
import pickle
import numpy as np
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Define the path to the model
MODEL_DIR = "D:\\Projects\\ML Projects\\Inappropriate Comment Scanner\\Dataset\\saved model"
logger.info(f"Model directory: {MODEL_DIR}")

# Function to clean text (simpler version)
def clean_text(text):
    """Clean the text for analysis"""
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text

# Function to analyze text directly
def analyze_text_direct(text, model, vectorizer, label_cols):
    """Analyze text for toxicity directly without using stored vectorizer"""
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Try to fit the vectorizer on the fly if needed
        if not hasattr(vectorizer, 'vocabulary_'):
            logger.info("Fitting vectorizer on the fly with the provided text")
            vectorizer.fit([cleaned_text])
        
        # Get features
        features = vectorizer.transform([cleaned_text])
        
        # Make predictions
        predictions = model.predict(features)[0]
        
        # Create results dictionary
        results = {}
        for i, label in enumerate(label_cols):
            results[label] = bool(predictions[i])
        
        # Add overall toxicity
        results['is_toxic'] = any(results.values())
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "predictions": results,
            "word_scores": {}  # Skip word scoring for simplicity
        }
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return {"error": str(e)}

# Try a fallback approach if vectorizer isn't fitted
def direct_classification(text):
    """Direct classification using simple keyword matching"""
    text = text.lower()
    
    # Define toxic keywords for each category
    toxic_keywords = {
        'toxic': ['idiot', 'stupid', 'dumb', 'moron', 'hate', 'fuck', 'shit', 'ass', 'jerk'],
        'severe_toxic': ['fuck', 'fucking', 'kill', 'die', 'murder'],
        'obscene': ['fuck', 'shit', 'ass', 'dick', 'pussy', 'cock', 'bitch'],
        'threat': ['kill', 'die', 'murder', 'hunt', 'find you'],
        'insult': ['idiot', 'stupid', 'dumb', 'ugly', 'fat', 'loser', 'moron'],
        'identity_hate': ['nigger', 'fag', 'gay', 'jew', 'muslim', 'islam', 'queer', 'retard']
    }
    
    # Check each category
    results = {}
    for category, keywords in toxic_keywords.items():
        results[category] = any(keyword in text for keyword in keywords)
    
    # Add overall toxicity
    results['is_toxic'] = any(results.values())
    
    return {
        "original_text": text,
        "cleaned_text": clean_text(text),
        "predictions": results,
        "word_scores": {}
    }

# HTML template as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inappropriate Comment Scanner (Fixed Version)</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            margin-bottom: 10px;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            margin-right: 10px;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            display: none;
        }
        .toxic {
            color: red;
            font-weight: bold;
        }
        .non-toxic {
            color: green;
            font-weight: bold;
        }
        .loading {
            display: none;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Inappropriate Comment Scanner (Fixed Version)</h1>
    
    <div>
        <p>Enter a comment to analyze for toxic content:</p>
        <textarea id="comment-input" placeholder="Enter your comment here..."></textarea>
        <div>
            <button id="analyze-btn">Analyze Comment</button>
            <button id="clear-btn">Clear</button>
        </div>
        
        <div class="loading" id="loading">Analyzing...</div>
        
        <div class="result" id="result">
            <div><strong>Overall:</strong> <span id="overall-result"></span></div>
            <div style="margin-top: 10px;"><strong>Categories:</strong></div>
            <div id="categories-result" style="margin-left: 15px;"></div>
            <div id="error-message" style="color: red; margin-top: 10px;"></div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const commentInput = document.getElementById('comment-input');
            const analyzeBtn = document.getElementById('analyze-btn');
            const clearBtn = document.getElementById('clear-btn');
            const resultDiv = document.getElementById('result');
            const overallResult = document.getElementById('overall-result');
            const categoriesResult = document.getElementById('categories-result');
            const loadingDiv = document.getElementById('loading');
            const errorMessage = document.getElementById('error-message');
            
            analyzeBtn.addEventListener('click', async function() {
                const text = commentInput.value.trim();
                
                if (!text) {
                    alert('Please enter a comment to analyze');
                    return;
                }
                
                // Show loading
                loadingDiv.style.display = 'block';
                resultDiv.style.display = 'none';
                errorMessage.textContent = '';
                
                try {
                    // Send request to API
                    const response = await fetch('/analyze_fallback', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Update overall result
                    const isToxic = data.predictions.is_toxic;
                    overallResult.textContent = isToxic ? 'TOXIC' : 'NON-TOXIC';
                    overallResult.className = isToxic ? 'toxic' : 'non-toxic';
                    
                    // Update categories
                    categoriesResult.innerHTML = '';
                    let toxicCategories = [];
                    
                    for (const [category, value] of Object.entries(data.predictions)) {
                        if (category !== 'is_toxic' && value === true) {
                            toxicCategories.push(category.replace('_', ' '));
                        }
                    }
                    
                    if (toxicCategories.length > 0) {
                        categoriesResult.textContent = toxicCategories.join(', ');
                    } else {
                        categoriesResult.textContent = 'None';
                    }
                    
                    // Show result
                    resultDiv.style.display = 'block';
                } catch (error) {
                    console.error('Error:', error);
                    errorMessage.textContent = 'Error: ' + error.message;
                    resultDiv.style.display = 'block';
                } finally {
                    // Hide loading
                    loadingDiv.style.display = 'none';
                }
            });
            
            clearBtn.addEventListener('click', function() {
                commentInput.value = '';
                resultDiv.style.display = 'none';
            });
        });
    </script>
</body>
</html>
"""

# Route for the home page
@app.route('/')
def home():
    return HTML_TEMPLATE

# API endpoint for analyzing text
@app.route('/analyze_fallback', methods=['POST'])
def analyze_fallback():
    # Get the text from the request
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    logger.info(f"Analyzing text: {text[:50]}...")
    
    # Try to load models and analyze
    try:
        # Try to load models
        model_path = os.path.join(MODEL_DIR, 'svm_tfidf_model.pkl')
        vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
        labels_path = os.path.join(MODEL_DIR, 'label_cols.pkl')
        
        if (os.path.exists(model_path) and os.path.exists(vectorizer_path) and 
            os.path.exists(labels_path)):
            # Load models
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                
                with open(labels_path, 'rb') as f:
                    label_cols = pickle.load(f)
                
                # Try direct analysis
                results = analyze_text_direct(text, model, vectorizer, label_cols)
                if "error" not in results:
                    return jsonify(results)
                else:
                    logger.warning(f"Direct analysis failed: {results['error']}, falling back to keyword matching")
            except Exception as e:
                logger.error(f"Error loading or using models: {e}")
        else:
            logger.warning("One or more model files not found")
        
        # Fall back to direct classification
        logger.info("Using fallback classification method")
        results = direct_classification(text)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Exception in analysis: {e}")
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == '__main__':
    logger.info("Starting web server...")
    logger.info("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)