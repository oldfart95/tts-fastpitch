import torch
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel

# Load the pre-trained FastPitch model
fastpitch = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
# Load the pre-trained HiFi-GAN vocoder
hifigan = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")

# Text to synthesize
text = "Hello! This is a test of the FastPitch text to speech system."

# Generate speech
parsed = fastpitch.parse(text)
spectrogram = fastpitch.generate_spectrogram(tokens=parsed)
audio = hifigan.convert_spectrogram_to_audio(spec=spectrogram)

# Save the audio to a file
import soundfile as sf
sf.write("output.wav", audio.to("cpu").detach().numpy()[0], 22050)
print("Audio file has been generated as 'output.wav'")
