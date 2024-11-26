# FastPitch Text-to-Speech Web Interface

A simple yet powerful web interface for text-to-speech synthesis using NVIDIA's FastPitch model. This application allows users to input text and generate high-quality speech audio through a clean web interface.

## Features

- Clean web interface for text input
- High-quality speech synthesis using FastPitch
- Real-time audio generation and playback
- Dockerized deployment for easy setup

## Quick Start

1. Make sure you have Docker and Docker Compose installed
2. Clone this repository
3. Run the application:
   ```bash
   docker-compose up --build
   ```
4. Open your browser and navigate to `http://localhost:8502`

## Technical Stack

- Flask: Web framework
- PyTorch: Deep learning framework
- NVIDIA FastPitch: Text-to-speech model
- Docker: Containerization

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
