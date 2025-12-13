# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code
COPY main.py .

# Create a writable cache directory for Hugging Face to store the model
# (Crucial step to avoid permission errors on HF Spaces)
RUN mkdir -p /app/cache
ENV HF_HOME=/app/cache
RUN chmod -R 777 /app/cache

# Expose the port
EXPOSE 7860

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]