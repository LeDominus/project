import os
import sys
from flask import Flask, render_template, jsonify, request, session # type: ignore

#* Добавляем корневую папку проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import analyze_cluster, analyze_sentiment, process_text, search_words_in_text

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '../templates'), 
            static_folder=os.path.join(os.path.dirname(__file__), '../static'))

app.secret_key = ("b9S4g7kO!LpR@qWzA3$GfVxE&hTjUeY")

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
            #* Анализ текста
            cluster_result = analyze_cluster(file_path)
            sentiment_result = analyze_sentiment(file_path)
            topics_result = process_text(file_path)           
            
            #* Сохраняю путь к файлу в сессии
            session['file_path'] = file_path

            #* Поиск слова
            search_word = request.form.get("search_word")
            word_search_result = None
            
            if search_word:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                word_search_result = search_words_in_text(file_content, search_word)
        except Exception as e:
            return render_template("index.html", cluster_result=f"Ошибка при анализе: {str(e)}", sentiment_result=None, word_search_result=None)
        finally:
            #* Удаление временного файла
            if os.path.exists(file_path):
                os.remove(file_path)

        return render_template("index.html", 
                               cluster_result=cluster_result,
                               sentiment_result=sentiment_result,
                               word_search_result=word_search_result,
                               topics_result=topics_result)

@app.route('/search', methods=['POST'])
def search_keywords():
    data = request.get_json()
    user_queries = data.get('user_query', [])
    file_path = "C:\\Users\\User-Максим\\Desktop\\LDA.docx"
    
    #* Обработка каждого ключевого слова
    results = []
    for query in user_queries:
        sentences_with_keyword = search_words_in_text(file_path, query)
        results.extend(sentences_with_keyword)
    
    return jsonify({'matches': results})



if __name__ == '__main__':
    app.run(debug=True)
