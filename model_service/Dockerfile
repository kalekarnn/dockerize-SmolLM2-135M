FROM python:3.9-slim

WORKDIR /app

# Install git for huggingface
RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY app.py .
COPY model.py .
COPY checkpoint_5050.pt ./checkpoint_model_5050.pt

EXPOSE 5000

CMD ["python", "app.py"] 