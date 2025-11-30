import os
import re
import torch
import pdfplumber
import textwrap
import traceback
import urllib.request
import warnings
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from fpdf import FPDF
from transformers import (
    PegasusTokenizer, PegasusForConditionalGeneration,
    LEDTokenizer, LEDForConditionalGeneration, LEDTokenizerFast
)

# Suppress Hugging Face warnings and general Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# --- Constants ---
# Ensure these paths correctly point to your fine-tuned model folders
FINE_TUNED_PEGASUS_MODEL_PATH = "./pegasus-finetuned-legal"
FINE_TUNED_LED_MODEL_PATH = "./led-finetuned-legal"
UPLOAD_FOLDER = "uploads"
SAVE_FOLDER = "saved" # Folder where summaries will be stored
MAX_LED_INPUT_LENGTH = 2048 # Max token length for LED model input

# --- Global device setting ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE.upper()}")

# --- Flask setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVE_FOLDER'] = SAVE_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

# --- Model Loading ---
print("🔄 Loading models...")
try:
    pegasus_tokenizer = PegasusTokenizer.from_pretrained(FINE_TUNED_PEGASUS_MODEL_PATH)
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(FINE_TUNED_PEGASUS_MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading fine-tuned Pegasus model: {e}. Falling back to default 'google/pegasus-xsum'.")
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

try:
    led_tokenizer = LEDTokenizerFast.from_pretrained(FINE_TUNED_LED_MODEL_PATH)
except Exception as e:
    print(f"⚠️ LEDTokenizerFast not available or failed ({e}). Falling back to LEDTokenizer.")
    led_tokenizer = LEDTokenizer.from_pretrained(FINE_TUNED_LED_MODEL_PATH)

led_model = LEDForConditionalGeneration.from_pretrained(FINE_TUNED_LED_MODEL_PATH)

# Move models to device and convert to half-precision if on GPU (saves VRAM)
pegasus_model.to(DEVICE)
led_model.to(DEVICE)

if DEVICE == "cuda":
    print("🚀 Converting models to float16 (half-precision) for GPU performance...")
    pegasus_model.half()
    led_model.half()
else:
    print("⚠️ Running on CPU. Float16 conversion skipped. Memory usage might be high for large inputs.")


# ------------------ Text Extraction & Cleaning ------------------

def clean_text(text):
    # Remove non-printable characters, consolidate newlines and spaces.
    # Allows common Latin characters and Devanagari (\u0900-\u097F) for Indian legal texts.
    cleaned_text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0900-\u097F]+', ' ', text)
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text) # Consolidate multiple newlines
    cleaned_text = re.sub(r'[ ]+', ' ', cleaned_text) # Consolidate multiple spaces
    return cleaned_text.strip()

def extract_text_from_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        text = clean_text(text)
        print("\n[DEBUG] Extracted text sample (first 500 chars):\n", text[:500], "\n---\n")
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF '{path}': {e}")
        traceback.print_exc() # Print full traceback for debugging
        return ""

def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        text = clean_text(text)
        print("\n[DEBUG] Extracted text sample (first 500 chars):\n", text[:500], "\n---\n")
        return text
    except Exception as e:
        print(f"❌ Error reading TXT file '{path}': {e}")
        traceback.print_exc() # Print full traceback for debugging
        return ""

# ------------------ Summary Formatting (Kept simple as per original `format_summary` function) ------------------

def format_summary(summary):
    # Splits summary into sentences and groups them into paragraphs for readability.
    sentences = re.split(r'(?<=[.!?]) +', summary)
    paragraphs = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    return "📝 Summary of Legal Document\n\n" + '\n\n'.join(paragraphs)

# ------------------ Summarization Logic (Now uses token count for model selection) ------------------

def summarize_text(text):
    text = text.strip()

    if DEVICE == "cuda":
        torch.cuda.empty_cache() # Clear CUDA cache to free up VRAM
        print("Cleared CUDA cache.")

    # Temporarily tokenize with LED tokenizer to get the actual token length.
    # This determines which model to use.
    temp_inputs_for_token_count = led_tokenizer(text, truncation=True, max_length=MAX_LED_INPUT_LENGTH, return_tensors="pt")
    current_token_count = temp_inputs_for_token_count["input_ids"].shape[1]

    # Pegasus-xsum typically has a max input length of 512 tokens.
    PEGASUS_TOKEN_THRESHOLD = 512 

    if current_token_count <= PEGASUS_TOKEN_THRESHOLD:
        print(f"🧠 Short text detected ({current_token_count} tokens). Using Pegasus model...")
        
        inputs = pegasus_tokenizer(
            text,
            return_tensors="pt",
            max_length=PEGASUS_TOKEN_THRESHOLD, # Truncate input to Pegasus's max length if needed
            truncation=True,
            padding="longest" # Pads to the longest sequence in the batch (or model max length if single input)
        ).to(DEVICE)

        summary_ids = pegasus_model.generate(
            **inputs, # Unpack input_ids and attention_mask
            max_length=150, # Max length of generated summary
            min_length=40,  # Min length of generated summary
            length_penalty=2.0, # Encourages longer summaries
            num_beams=4,    # Number of beams for beam search (higher means better quality, slower)
            early_stopping=True, # Stop generation when all hypotheses have met stopping criteria
            repetition_penalty=2.5, # Penalize repetition
            no_repeat_ngram_size=3, # Don't repeat n-grams of this size
            do_sample=False # Use deterministic beam search
        )
        summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    else: # If token count is greater than Pegasus's threshold, use LED
        print(f"📚 Long text detected ({current_token_count} tokens). Using LED model with max input length {MAX_LED_INPUT_LENGTH}...")
        
        # Re-tokenize for LED model, ensuring padding to max_length for consistent tensor shapes
        inputs = led_tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LED_INPUT_LENGTH,
            truncation=True,
            padding="max_length"
        ).to(DEVICE)

        # Global attention mask: set global attention on the first token (or other strategic tokens)
        global_attention_mask = torch.zeros_like(inputs["input_ids"]).to(DEVICE)
        global_attention_mask[:, 0] = 1 # Global attention on the first token (CLS token)

        summary_ids = led_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], # Corrected from attention_attention_mask
            global_attention_mask=global_attention_mask,
            num_beams=4,
            max_length=400, # Max length of generated summary
            min_length=100, # Min length of generated summary
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False
        )
        summary = led_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return format_summary(summary)

# ------------------ Save Functions ------------------

def save_as_txt(filename, summary_text):
    try:
        save_path = os.path.join(app.config['SAVE_FOLDER'], filename)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"✅ TXT saved as: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ Error saving TXT file '{filename}': {e}")
        traceback.print_exc()
        return None

def save_as_pdf(filename, summary_text):
    pdf = FPDF()
    pdf.add_page()
    # Path to the font file (DejaVuSans.ttf must be in the same directory as backend.py or accessible)
    font_path = "DejaVuSans.ttf" 
    
    # Attempt to download font if not found (for better Unicode support, e.g., Devanagari)
    if not os.path.exists(font_path):
        print("📥 Downloading DejaVuSans.ttf...")
        url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf" # Use raw URL
        try:
            urllib.request.urlretrieve(url, font_path)
            print("Font downloaded successfully.")
        except Exception as e:
            print(f"❌ Error downloading font: {e}. PDF might not render Indian scripts correctly.")
            print("Please ensure you have internet access or download DejaVuSans.ttf manually to the script's directory.")
            font_path = None # Indicate that custom font is not loaded

    if font_path and os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    else:
        print("⚠️ Using default PDF font (might not support all characters).")
        pdf.set_font("Arial", size=12) # Fallback to a standard font

    # Wrap text to fit PDF page width
    wrapped_summary = textwrap.fill(summary_text, width=100) # Adjust width as needed for your PDF layout

    for line in wrapped_summary.split('\n'):
        pdf.multi_cell(0, 10, line) # 0 for width means it will take the full page width
    
    try:
        save_path = os.path.join(app.config['SAVE_FOLDER'], filename)
        pdf.output(save_path)
        print(f"✅ PDF saved as: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ Error saving PDF file '{filename}': {e}")
        traceback.print_exc()
        return None


# ------------------ Flask Routes ------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/summary')
def summary_page():
    # This route just serves the HTML page; actual summary content is handled by JS from localStorage
    return render_template('summary.html')

@app.route('/saved')
def saved():
    return render_template('saved.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/summarize', methods=['POST'])
def summarize_api():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path) # Save uploaded file temporarily

        text = ""
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload .pdf or .txt.'}), 400

        # Check if extracted text is valid for summarization
        if not text.strip() or len(text.strip().split()) < 20:
            return jsonify({'error': 'Extracted text is empty or too short for summarization (less than 20 words).'}), 400

        summary = summarize_text(text) # Get the formatted summary text
        
        # Determine base name for saving summary files
        base_name = os.path.splitext(filename)[0]
        
        # Save summary as TXT
        txt_filename = f"{base_name}_summary.txt"
        save_as_txt(txt_filename, summary)

        # Save summary as PDF
        pdf_filename = f"{base_name}_summary.pdf"
        save_as_pdf(pdf_filename, summary)

        # Clean up the uploaded temporary file
        os.remove(file_path)
        print(f"Cleaned up temporary file: {file_path}")

        # Return the summary text and the PDF filename (for download on the summary page)
        return jsonify({
            'summary': summary,
            'filename': pdf_filename # Frontend will use this for the download link
        })

    except Exception as e:
        print(f"Error in summarize_api: {e}")
        traceback.print_exc() # Print full traceback for easier debugging
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

@app.route('/list_summaries')
def list_summaries():
    """Returns a list of all saved summary filenames (both .txt and .pdf)."""
    try:
        files = [f for f in os.listdir(app.config['SAVE_FOLDER']) if f.endswith(('.txt', '.pdf'))]
        # Sort files by modification time, newest first (optional but helpful)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['SAVE_FOLDER'], x)), reverse=True)
        return jsonify(files)
    except Exception as e:
        print(f"Error listing summaries: {e}")
        traceback.print_exc()
        return jsonify([])

@app.route('/summaries/<path:filename>')
def download_summary(filename):
    """Serves a specific saved summary file for download."""
    try:
        return send_from_directory(app.config['SAVE_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        traceback.print_exc()
        return "Error serving file.", 500

if __name__ == '__main__':
    app.run(debug=True) 


