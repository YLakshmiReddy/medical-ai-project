# D:\medical_ai_project\Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Set the environment variable for caching to avoid symlink warnings on some systems
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HOME /app/.cache/huggingface

# Removed TRANSFORMERS_BITSANDBYTES_CPU_COMPATIBILITY for Colab path.
# This Dockerfile is for Render/HuggingFace if you try Docker again.
# For Colab, Dockerfile is not used directly.

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 10000 

# Command to run the application
CMD ["bash", "entrypoint.sh"] # Use bash to run the shell script