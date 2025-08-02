# D:\medical_ai_project\Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Install other packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for caching to avoid symlink warnings on some systems
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HOME /app/.cache/huggingface # Ensure cache is within the /app directory

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application (use entrypoint.sh for more complex commands)
# CMD ["python", "app.py"] # This would directly run app.py, but we'll use entrypoint.sh