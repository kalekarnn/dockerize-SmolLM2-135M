import requests
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

MODEL_SERVICE_URL = "http://model-service:5000/generate"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Text Generator</title>
    <style>
        body { max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; height: 100px; }
        .output { white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Text Generator</h1>
    <form method="POST">
        <p>
            <label>Enter your prompt:</label><br>
            <textarea name="prompt" required>{{ prompt }}</textarea>
        </p>
        <p>
            <label>Maximum Length:</label>
            <input type="number" name="max_length" value="{{ max_length }}" min="10" max="500">
        </p>
        <button type="submit">Generate</button>
    </form>
    {% if generated_text %}
    <h2>Generated Text:</h2>
    <div class="output">{{ generated_text }}</div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    generated_text = ""
    prompt = ""
    max_length = 100

    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        max_length = int(request.form.get("max_length", 100))

        try:
            response = requests.post(
                MODEL_SERVICE_URL, json={"prompt": prompt, "max_length": max_length}
            )
            result = response.json()
            generated_text = result.get("generated_text", "")
        except Exception as e:
            generated_text = f"Error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        prompt=prompt,
        max_length=max_length,
        generated_text=generated_text,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
