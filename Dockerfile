# Use the official image with GPU support
FROM ultralytics/ultralytics:latest

WORKDIR /app

# We don't need to install system libraries (libgl1) anymore, 
# because this image already has them!

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
