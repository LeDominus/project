import os
import sys
from flask import Flask, render_template, jsonify, request # type: ignore

# Добавляем корневую папку проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import analyze_cluster, analyze_sentiment, process_text

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '../templates'), 
            static_folder=os.path.join(os.path.dirname(__file__), '../static'))

@app.route("/")
def index():
    return render_template("index.html", cluster_result=None, sentiment_result=None)

@app.route("/analyze", methods=['POST'])
def analyze_text():
    if 'file' not in request.files:
        return render_template("index.html", cluster_result="Ошибка: нет файла", sentiment_result=None)
    file = request.files['file']

    if file.filename == '':
        return render_template("index.html", cluster_result="Ошибка: файл не выбран", sentiment_result=None)

    if file:
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)

        cluster_result = analyze_cluster(file_path)
        sentiment_result = analyze_sentiment(file_path)
        process_text(file_path)

        return render_template("index.html", 
                               cluster_result=cluster_result,
                               sentiment_result=sentiment_result)

if __name__ == '__main__':
    app.run(debug=True)
