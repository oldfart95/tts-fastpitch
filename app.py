from flask import Flask, request, send_file, render_template_string, session
import torch
import soundfile as sf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
import io
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Text-to-Speech with FastPitch</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        textarea {
            width: 100%;
            height: 100px;
            margin: 10px 0;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .audio-container {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Text-to-Speech</h1>
        <form action="/generate" method="post">
            <textarea name="text" placeholder="Enter text to convert to speech">{{ current_text if current_text else "Hello, this is a test of the FastPitch text to speech system." }}</textarea>
            <br>
            <button type="submit">Generate Speech</button>
        </form>
        {% if audio_path %}
        <div class="audio-container">
            <h3>Generated Audio:</h3>
            <audio controls>
                <source src="{{ audio_path }}?text={{ current_text|urlencode }}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            <p><a href="{{ audio_path }}?text={{ current_text|urlencode }}" download="generated_speech.wav">Download audio file</a></p>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

# Load models
spec_generator = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
vocoder = HifiGanModel.from_pretrained("nvidia/tts_hifigan")

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    session['current_text'] = text
    
    return render_template_string(
        HTML_TEMPLATE,
        audio_path='/audio',
        current_text=text
    )

@app.route('/audio')
def serve_audio():
    text = request.args.get('text', "Hello, this is a test.")
    
    with torch.no_grad():
        parsed = spec_generator.parse(text)
        spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
    
    # Convert to wav file
    audio_data = audio.to("cpu").detach().numpy()[0]
    
    # Save to memory buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, 22050, format='WAV')
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='generated_speech.wav'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)
