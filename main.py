from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import re
import os
import logging
import json
from typing import Dict, Tuple
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def extract_article(self, url: str) -> Tuple[str, str]:
        try:
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            title = ''
            title_tag = soup.find('h1')
            if title_tag:
                title = title_tag.get_text().strip()

            # Try different selectors for article content
            article = None
            for selector in ['article', 'div[class*="article"]', 'div[class*="content"]', 
                           'div[class*="post"]', 'main']:
                article = soup.select_one(selector)
                if article:
                    break

            if not article:
                raise ValueError("Could not find article content")
            
            # Remove unwanted elements
            unwanted_tags = ['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'iframe', 'form', 'button']
            for tag in unwanted_tags:
                for element in article.find_all(tag):
                    element.decompose()

            text = ' '.join(p.get_text().strip() for p in article.find_all('p'))
            text = re.sub(r'\s+', ' ', text).strip()

            return title, text

        except Exception as e:
            logging.error(f"Error extracting from {url}: {str(e)}")
            return "", ""

class GroqAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def analyze_text(self, title: str, text: str) -> Dict:
        try:
            # Prepare the prompt for analysis
            prompt = f"""Analyze the following article titled "{title}":

{text}

Provide a comprehensive analysis including:
1. A concise summary (2-3 sentences)
2. Key points (3-5 points)
3. Sentiment analysis (positive/negative/neutral with explanation)
4. Main topics discussed
5. Writing style analysis
6. Target audience assessment

Format the response as JSON with these exact keys:
- summary
- key_points (array)
- topics (array)
- writing_style"""

            # Make request to GROQ API
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "llama-3.3-70b-versatile",  # Using Mixtral model
                    "messages": [
                        {"role": "system", "content": "You are an expert text analyzer. Provide detailed, accurate analysis in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"}
                }
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if 'choices' not in response_data or not response_data['choices']:
                raise ValueError("Invalid response format from GROQ API")
                
            analysis_text = response_data['choices'][0]['message']['content']
            
            try:
                # Parse the JSON response with error handling
                analysis = json.loads(analysis_text)
                
                # Validate expected keys are present
                expected_keys = {'summary', 'key_points', 'sentiment', 'topics', 
                               'writing_style', 'target_audience'}
                if not all(key in analysis for key in expected_keys):
                    raise ValueError("Missing required keys in analysis response")
                
                return analysis
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse GROQ response as JSON: {e}")
                raise ValueError(f"Invalid JSON response from GROQ: {e}")

        except requests.RequestException as e:
            logging.error(f"GROQ API request failed: {str(e)}")
            return {
                "error": "Failed to connect to GROQ API",
                "details": str(e)
            }
        except Exception as e:
            logging.error(f"Error in GROQ analysis: {str(e)}")
            return {
                "error": "Failed to analyze text",
                "details": str(e)
            }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        url = request.form.get('url')
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format. URL must start with http:// or https://'}), 400
        
        # Initialize scraper and analyzer
        scraper = WebScraper()
        analyzer = GroqAnalyzer(GROQ_API_KEY)

        # Extract content
        title, text = scraper.extract_article(url)
        if not text:
            return jsonify({'error': 'Failed to extract article content'}), 400

        # Analyze with GROQ
        analysis = analyzer.analyze_text(title, text)
        
        # Check if there was an error in analysis
        if 'error' in analysis:
            return jsonify(analysis), 500
            
        return jsonify(analysis)

    except Exception as e:
        logging.error(f"Error in analysis endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if not GROQ_API_KEY:
        raise ValueError("GROQ API key not found in environment variables")
    app.run(debug=True)