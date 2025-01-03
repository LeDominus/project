import os
import sys
import fitz
from quart import Quart, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import asyncio

#* Добавляем корневую папку проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model_studybook import calculate_similarity, classify_style, analyze_coherence, analyze_structure, check_grammar

app = Quart(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), '../templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '../static'))

app.secret_key = ("b9S4g7kO!LpR@qWzA3$GfVxE&hTjUeY")

UPLOAD_FOLDER = 'temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#* Функция для извлечения текста из PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        file = await request.files  # Получаем файлы из запроса асинхронно

        if 'file' not in file:
            return await render_template("index.html", error='Ошибка: нет файла')

        uploaded_file = file['file']

        if uploaded_file.filename == '':
            return await render_template("index.html", error='Ошибка: файл не выбран')

        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            await uploaded_file.save(file_path)  # Асинхронно сохраняем файл

            try:
                text = extract_text_from_pdf(file_path)

                original_text = 'C:\\Programming\\Python\\Python developing\\python_basics\\Project_BERT\\PDF_учебники\\201_Stat.pdf'

                style_result = classify_style(text)
                coherence_result = await analyze_coherence(text)
                structure_result = analyze_structure(original_text, text)
                
            except Exception as e:
                print(f"Ошибка анализа: {str(e)}")
                return await render_template("index.html", error=f'Ошибка при анализе: {str(e)}')

            return await render_template("index.html",                                       
                                         style_result=style_result,
                                         coherence_result=f"{coherence_result:.2f}",
                                         structure_result=structure_result)

    return await render_template("index.html")

@app.route('/favicon.ico')
async def favicon():
    return await send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
