# app_fixed_enhanced.py - Fixed version with sample text selector

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

# HTML template with sample text selector
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inappropriate Comment Scanner</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        .examples {
            margin-bottom: 15px;
        }
        .examples select {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
            font-size: 14px;
            width: 100%;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            box-sizing: border-box;
            margin-bottom: 15px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        .button-secondary {
            background-color: #f1f1f1;
            color: #333;
        }
        .button-secondary:hover {
            background-color: #e0e0e0;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
            display: none;
        }
        .result-heading {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 18px;
        }
        .toxic {
            color: #d32f2f;
            font-weight: bold;
        }
        .non-toxic {
            color: #388e3c;
            font-weight: bold;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .spinner {
            width: 40px;
            height: 40px;
            margin: 0 auto;
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left-color: #4CAF50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .category-item {
            margin-bottom: 5px;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 14px;
            color: #777;
        }
    </style>
</head>
<body>
    <h1>Inappropriate Comment Scanner</h1>
    
    <div class="container">
        <div class="examples">
            <label for="example-select">Examples:</label>
            <select id="example-select">
                <option value="">Select an example...</option>
                <option value="This is a normal comment about the weather.">Normal comment about weather</option>
                <option value="I appreciate your thoughtful response to my question.">Polite appreciation</option>
                <option value="You're such a moron, only an idiot would think that.">Insulting comment</option>
                <option value="Go kill yourself, nobody likes you anyway.">Threatening comment</option>
                <option value="People like you shouldn't be allowed to vote.">Subtle toxic comment</option>
                <option value="This product is terrible and the company is a scam.">Negative but not toxic</option>
                <option value="I completely disagree with everything you said.">Disagreement</option>
                <option value="F*** off with that nonsense, you stupid jerk.">Explicit language</option>
                <option value="Women are too emotional to be in leadership positions.">Gender bias</option>
                <option value="All people from that country are criminals.">Xenophobic comment</option>
            </select>
        </div>
        
        <textarea id="comment-input" placeholder="Enter your comment here..."></textarea>
        
        <div>
            <button id="analyze-btn">Analyze Comment</button>
            <button id="clear-btn" class="button-secondary">Clear</button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing comment...</p>
        </div>
        
        <div class="result" id="result">
            <div class="result-heading">Analysis Result</div>
            
            <div>
                <strong>Overall:</strong> 
                <span id="overall-result"></span>
            </div>
            
            <div style="margin-top: 10px;">
                <strong>Categories:</strong>
                <div id="categories-result" style="margin-left: 15px;"></div>
            </div>
            
            <div id="error-message" style="color: red; margin-top: 10px;"></div>
        </div>
    </div>
    
    <div class="footer">
        <p>This tool analyzes text for toxic content. Words highlighted in red may contribute to toxicity.</p>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const commentInput = document.getElementById('comment-input');
            const analyzeBtn = document.getElementById('analyze-btn');
            const clearBtn = document.getElementById('clear-btn');
            const exampleSelect = document.getElementById('example-select');
            const resultDiv = document.getElementById('result');
            const overallResult = document.getElementById('overall-result');
            const categoriesResult = document.getElementById('categories-result');
            const loadingDiv = document.getElementById('loading');
            const errorMessage = document.getElementById('error-message');
            
            // Handle example selection
            exampleSelect.addEventListener('change', function() {
                if (this.value) {
                    commentInput.value = this.value;
                }
            });
            
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
                        toxicCategories.forEach(category => {
                            const categoryItem = document.createElement('div');
                            categoryItem.className = 'category-item';
                            categoryItem.textContent = category;
                            categoriesResult.appendChild(categoryItem);
                        });
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
                exampleSelect.value = '';
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