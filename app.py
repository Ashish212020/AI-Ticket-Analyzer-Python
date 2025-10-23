from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load the NLP pipelines
# We use "zero-shot-classification" to define our own categories on the fly
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") 

# A standard sentiment model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# A summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        
        # Ensure 'text' key exists
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' key in request body"}), 400
            
        text_to_analyze = data['text']
        
        # --- 1. Classification ---
        # Define the categories you want to classify into
        candidate_labels = ["Billing", "Technical Support", "Bug Report", "Feature Request", "General Feedback"]
        classification_result = classifier(text_to_analyze, candidate_labels)
        
        # --- 2. Sentiment Analysis ---
        sentiment_result = sentiment_analyzer(text_to_analyze)
        
        # --- 3. Summarization ---
        # Ensure text is not too short for summarization
        if len(text_to_analyze.split()) > 30: # Only summarize if text is > 30 words
            summary_result = summarizer(text_to_analyze, max_length=50, min_length=15, do_sample=False)
        else:
            summary_result = [{"summary_text": "Text too short to summarize."}]
            
        # --- Combine Results ---
        result = {
            "classification": {
                "label": classification_result['labels'][0],
                "score": classification_result['scores'][0]
            },
            "sentiment": {
                "label": sentiment_result[0]['label'],
                "score": sentiment_result[0]['score']
            },
            "summary": summary_result[0]['summary_text']
        }
        
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    # Run on port 5000
    app.run(debug=True, port=5000)