"""from flask import Flask, render_template, request
import yt_dlp
import whisper
import os

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")


# -------------------------
# Function to download audio from a link
# -------------------------
def download_from_link(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "link_audio.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return "link_audio.mp3"


# -------------------------
# Home Page
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')


# -------------------------
# Handle Upload or URL
# -------------------------
@app.route('/transcribe', methods=['POST'])
def transcribe():
    choice = request.form.get("choice")

    # -------------------------
    # Option 1 → Upload File
    # -------------------------
    if choice == "1":
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']
        if file.filename == '':
            return "Empty File"

        filepath = file.filename
        file.save(filepath)

        result = model.transcribe(filepath, fp16=False)

        # remove saved file
        os.remove(filepath)

        return f""
        <h3>Detected Language: {result["language"]}</h3>
        <pre>{result["text"]}</pre>
        ""

    # -------------------------
    # Option 2 → URL Link
    # -------------------------
    elif choice == "2":
        url = request.form.get("url")
        if not url:
            return "No URL provided"

        audio_file = download_from_link(url)
        result = model.transcribe(audio_file, fp16=False)

        # remove downloaded file
        os.remove(audio_file)

        return f""
        <h3>Detected Language: {result["language"]}</h3>
        <pre>{result["text"]}</pre>
        ""

    else:
        return "Invalid choice."


if __name__ == "__main__":
    app.run(debug=True, port=5001)""
from flask import Flask, render_template, request
import yt_dlp
import whisper
import os

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")


# -------------------------
# Function to download audio from a link
# -------------------------
def download_from_link(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "link_audio.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return "link_audio.mp3"


# -------------------------
# Home Page
# -------------------------
@app.route('/')
def index():
    # Render the page with no results initially
    return render_template('index.html', transcript=None, language=None)


# -------------------------
# Handle Upload or URL
# -------------------------
@app.route('/transcribe', methods=['POST'])
def transcribe():
    choice = request.form.get("choice")
    transcript_text = ""
    detected_lang = ""
    error_msg = None

    try:
        # -------------------------
        # Option 1 → Upload File
        # -------------------------
        if choice == "1":
            if 'file' not in request.files:
                error_msg = "No file part"
            else:
                file = request.files['file']
                if file.filename == '':
                    error_msg = "No selected file"
                else:
                    filepath = file.filename
                    file.save(filepath)
                    
                    # Transcribe
                    result = model.transcribe(filepath, fp16=False)
                    transcript_text = result["text"]
                    detected_lang = result["language"]

                    # Cleanup
                    os.remove(filepath)

        # -------------------------
        # Option 2 → URL Link
        # -------------------------
        elif choice == "2":
            url = request.form.get("url")
            if not url:
                error_msg = "No URL provided"
            else:
                audio_file = download_from_link(url)
                
                # Transcribe
                result = model.transcribe(audio_file, fp16=False)
                transcript_text = result["text"]
                detected_lang = result["language"]

                # Cleanup
                os.remove(audio_file)

        else:
            error_msg = "Invalid choice selected."

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"

    # Return the same HTML page, but inject the results or errors
    return render_template(
        'index.html', 
        transcript=transcript_text, 
        language=detected_lang, 
        error=error_msg
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)"""

from flask import Flask, render_template, request, jsonify
import yt_dlp
import whisper
import os
import nltk
from deep_translator import GoogleTranslator

# --- Hugging Face Imports for Paraphrasing ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = Flask(__name__)

# ------------------------------------------------
# 1. LOAD MODELS (Load once to save time)
# ------------------------------------------------
print("Loading Whisper Model...")
whisper_model = whisper.load_model("base")

print("Loading Paraphrase Model (Pegasus)...")
# Using PyTorch version to match Whisper's dependencies and save RAM
tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
paraphrase_pipeline = pipeline('text2text-generation', model=pegasus_model, tokenizer=tokenizer, truncation=True)

# Ensure NLTK data is available for sentence splitting
nltk.download('punkt', quiet=True)


# -------------------------
# Helper: Download Audio
# -------------------------
def download_from_link(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "link_audio.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "link_audio.mp3"


# -------------------------
# Route: Home
# -------------------------
@app.route('/')
def index():
    return render_template('index.html', transcript=None)


# -------------------------
# Route: Transcribe (Audio to Text)
# -------------------------
@app.route('/transcribe', methods=['POST'])
def transcribe():
    choice = request.form.get("choice")
    transcript_text = ""
    detected_lang = ""
    error_msg = None

    try:
        # Handle File Upload
        if choice == "1":
            if 'file' not in request.files:
                error_msg = "No file part"
            else:
                file = request.files['file']
                if file.filename == '':
                    error_msg = "No selected file"
                else:
                    filepath = file.filename
                    file.save(filepath)
                    result = whisper_model.transcribe(filepath, fp16=False)
                    transcript_text = result["text"]
                    detected_lang = result["language"]
                    os.remove(filepath)

        # Handle URL
        elif choice == "2":
            url = request.form.get("url")
            if not url:
                error_msg = "No URL provided"
            else:
                audio_file = download_from_link(url)
                result = whisper_model.transcribe(audio_file, fp16=False)
                transcript_text = result["text"]
                detected_lang = result["language"]
                os.remove(audio_file)
        else:
            error_msg = "Invalid choice."

    except Exception as e:
        error_msg = f"Error: {str(e)}"

    return render_template(
        'index.html', 
        transcript=transcript_text, 
        language=detected_lang, 
        error=error_msg
    )


# -------------------------
# Route: Translate
# -------------------------
@app.route('/translate_text', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        text = data.get('text')
        target_lang = data.get('target_lang')

        if not text or not target_lang:
            return jsonify({"error": "Missing data"}), 400

        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return jsonify({"result": translated, "type": "Translation"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Route: Paraphrase
# -------------------------
@app.route('/paraphrase_text', methods=['POST'])
def paraphrase_text_route():
    try:
        data = request.get_json()
        text = data.get('text')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Split text into sentences (Pegasus works best on single sentences)
        sentences = nltk.sent_tokenize(text)
        
        # Paraphrase each sentence
        # num_return_sequences=1 means we get 1 version per sentence
        paraphrased_sentences = [paraphrase_pipeline(sent, num_return_sequences=1, num_beams=5)[0]['generated_text'] for sent in sentences]
        
        final_text = ' '.join(paraphrased_sentences)
        
        return jsonify({"result": final_text, "type": "Paraphrased Text"})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5003)
