# SmolLM2-135M Text Generation Service

This project implements a text generation service using a custom SmolLM2-135M model architecture. It consists of two services:
- A model service that handles the text generation
- A web interface for interacting with the model

## Project Structure 
```
project/
├── model_service/
│ ├── app.py # Model service Flask application
│ ├── model.py # SmolLM2-135M model implementation
│ ├── Dockerfile # Model service container configuration
│ ├── requirements.txt # Model service dependencies
│ └── checkpoint_5050.pt # Model weights
├── app/
│ ├── app.py # Web interface Flask application
│ ├── Dockerfile # Web interface container configuration
│ └── requirements.txt # Web interface dependencies
├── docker-compose.yml # Docker services configuration
└── README.md # This file
```


## Prerequisites

- Docker and Docker Compose installed
- Python 3.9 or later (for local development)
- The model checkpoint file (`checkpoint_5050.pt`)

## Setup and Running

1. Make sure the model checkpoint is in the correct location:

Download the model checkpoint from [SmolLM2-135-model checkpoint_5050.pt](https://huggingface.co/spaces/kalekarnn/SmolLM2-135-model/resolve/main/checkpoint_5050.pt)
```
cp path/to/checkpoint_5050.pt model_service/checkpoint_5050.pt
```

2. Build and run the Docker containers:
```
docker-compose up --build
```

3. Access the web interface:
- Open your browser and go to `http://localhost:8000`
- Enter your prompt and adjust the maximum length as needed
- Click "Generate" to create text

## API Endpoints

### Model Service (port 5000)
- POST `/generate`
  - Request body:
    ```json
    {
        "prompt": "Your text prompt here",
        "max_length": 100
    }
    ```
  - Response:
    ```json
    {
        "generated_text": "Generated text response..."
    }
    ```

### Web Interface (port 8000)
- GET `/` - Web interface for text generation
- POST `/` - Handle form submission for text generation

## Development

To run the services separately for development:

1. Model Service:
```
cd model_service
pip install -r requirements.txt
python app.py
```

2. Web Interface:
```
cd app
pip install -r requirements.txt
python app.py
```
