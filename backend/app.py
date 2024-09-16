import os
import sys
from flask import Flask, render_template, jsonify, request # type: ignore

# Добавляем корневую папку проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import analyze_cluster, analyze_sentiment, find_words, process_text

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '../templates'), 
            static_folder=os.path.join(os.path.dirname(__file__), '../static'))

@app.route("/")
def index():
    return render_template("index.html", cluster_result=None, sentiment_result=None, word_search_result=None)

@app.route("/analyze", methods=['POST'])
def analyze_text():
    if 'file' not in request.files:
        return render_template("index.html", cluster_result="Ошибка: нет файла", sentiment_result=None, word_search_result=None)
    
    file = request.files['file']

    if file.filename == '':
        return render_template("index.html", cluster_result="Ошибка: файл не выбран", sentiment_result=None, word_search_result=None)

    if file:
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)

        try:
            cluster_result = analyze_cluster(file_path)
            sentiment_result = analyze_sentiment(file_path)
            process_text(file_path)  # Если функция process_text возвращает результаты, сохраните их в переменной
            
            search_word = request.form.get("search_word")
            word_search_result = None
            
            if search_word:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                word_search_result = find_words(file_content, search_word)
        except Exception as e:
            return render_template("index.html", cluster_result=f"Ошибка при анализе: {str(e)}", sentiment_result=None, word_search_result=None)
        finally:
            # Удаление временного файла
            if os.path.exists(file_path):
                os.remove(file_path)

        return render_template("index.html", 
                               cluster_result=cluster_result,
                               sentiment_result=sentiment_result,
                               word_search_result=word_search_result)

if __name__ == '__main__':
    app.run(debug=True)
