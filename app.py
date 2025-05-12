from flask import Flask, render_template, request
from models.bias_detection import detect_bias

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get text input from the user
        text = request.form["news_article"]
        
        # Call the bias detection model function
        result = detect_bias(text)
        
        # Render the result to the webpage
        return render_template("index.html", result=result, text=text)
    
    return render_template("index.html", result=None, text=None)

if __name__ == "__main__":
    app.run(debug=True)
