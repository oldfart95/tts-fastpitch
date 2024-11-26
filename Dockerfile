FROM nvcr.io/nvidia/nemo:23.06

# Install compatible versions of dependencies
RUN pip install numpy==1.22.2 && \
    pip install flask

# Copy the application files
WORKDIR /workspace/tts
COPY app.py .

# Expose the Flask port
EXPOSE 8501

# Run Flask
CMD ["python", "app.py"]
