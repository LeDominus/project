import re
import language_tool_python
import pdfplumber
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BartModel, BartTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from tqdm.asyncio import tqdm_asyncio
import torch.nn.functional as F


'''
СЕКЦИЯ ОБРАБОТКИ ДАННЫХ ПЕРЕД АНАЛИЗОМ ТЕКСТА
'''

#* Функция для конвертирования текста из PDF файла
def convert_text_from_pdf(file_path):
    full_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            full_text.append(page.extract_text())
    return '\n'.join(full_text)

#* Функции для работы с моделями BERT
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

#* Функция для получения эмбеддингов текста
async def get_embeddings(text):
    inputs = tokenizer_bart(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

#* Модель для классификации стиля

tokenizer_style = BertTokenizer.from_pretrained("bert-base-uncased")
model_style = BertForSequenceClassification.from_pretrained("any0019/text_style_classifier")

def classify_style(text):
    inputs = tokenizer_style(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_style(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    styles = ['Официально-деловой стиль', 'Художественный стиль', 'Научный стиль', 'Публицистический стиль', 'Разговорный стиль']
    return styles[predicted_class]

#* Функция для расчета схожести двух текстов
async def calculate_similarity(text1, text2):

    embedding1 = await get_embeddings(text1)
    embedding2 = await get_embeddings(text2)
    
    if torch.all(embedding1 == 0) or torch.all(embedding2 == 0):
        print("Предупреждение: один из векторов имеет нулевое значение.")
        return 0.0
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()

#* Функция для извлечения структуры из текста
def extract_structure(text, section_patterns):
    structure = []
    for pattern in section_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            structure.append(matches[0])
    return structure

#* Функция для анализа структуры текста
def analyze_structure(original_text, reference_text):
    section_patterns = [
        r'\b(введение|предисловие)\b',  
        r'\b(глава|лекция|тема)\s+\d+\b',  
        r'\bзаключение|вывод|итоги\b',  
        r'\b(список использованной литературы|список рекомендованной литературы)\b',
        r'\bсодержание|оглавление\b'
    ]
    
    original_structure = extract_structure(original_text, section_patterns)
    reference_structure = extract_structure(reference_text, section_patterns)

    if not reference_structure:
        print("Ошибка: отсутствуют разделы в эталонном тексте.")
        return 0.0

    if original_structure == reference_structure:
        return 1.0
    else:
        matching_sections = set(original_structure) & set(reference_structure)
        return len(matching_sections) / len(reference_structure) if len(reference_structure) > 0 else 0.0


#* Асинхронная функция для грамматической проверки текста
async def check_grammar(text):
    tool = language_tool_python.LanguageTool('ru')
    matches = await asyncio.to_thread(tool.check, text)
    if matches:
        errors = [{'message': match.message, 'error': match.context} for match in matches]
        return errors
    else:
        print('Ошибок не найдено')

#* Асинхронная функция для когерентного анализа текста

tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
model_bart = BartModel.from_pretrained('facebook/bart-large-mnli')

async def get_embeddings_async(text):
    inputs = tokenizer_bart(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

async def analyze_coherence(text):
    lines = text.split('\n')
    sections = [' '.join(lines[i:i+30]) for i in range(0, len(lines), 30)]
    
    if len(sections) < 2:
        print('Недостаточно секций для анализа когерентности.')
        return 0.0
    
    embeddings = await asyncio.gather(*[get_embeddings_async(section) for section in sections])
    
    coherence_score = []
    for i in range(1, len(embeddings)):
        similarity = torch.nn.functional.cosine_similarity(embeddings[i-1], embeddings[i], dim=-1)
        avg_similarity = similarity.mean().item()
        coherence_score.append(avg_similarity)
    
    if not coherence_score:
        print('Ошибка: не удалось рассчитать когерентность, отсутствуют данные для сравнения.')
        return 0.0

    average_coherence = sum(coherence_score) / len(coherence_score) if len(coherence_score) > 0 else 0.0
    print(f'Средний коэффициент когерентности: {average_coherence}')
    
    return average_coherence


#* Функция для отображения прогресса
async def show_loading(computation_task):
    progress_bar = tqdm_asyncio(total=1, desc='Обработка результатов', unit='task')
    result = await computation_task
    progress_bar.update(1)
    progress_bar.close()
    return result

#* Основная функция для запуска
async def main():
    file_path_original = 'C:\\Users\\User-Максим\\Downloads\\КФУ_статистика.pdf'
    file_path_reference = 'C:\\Programming\\Python\\Python developing\\python_basics\\Project_BERT\\PDF_учебники\\Проба.pdf'
    
    original_text = convert_text_from_pdf(file_path_original)
    reference_text = convert_text_from_pdf(file_path_reference)

    print('Проверка стиля...')
    style_original_text = classify_style(original_text)
    
    similarity_score = await calculate_similarity(original_text, reference_text)
    
    coherence_task = asyncio.create_task(analyze_coherence(reference_text))
    coherence_score = await show_loading(coherence_task)
    
    structure_task = asyncio.create_task(asyncio.to_thread(analyze_structure, original_text, reference_text))
    structure_score = await show_loading(structure_task)
    
    result = {
        "style_result": style_original_text,
        "similarity_score_result": similarity_score,
        "coherence_result": coherence_score,
        "structure_result": structure_score,
    }
    
    return result

# СТАРТУЕМ
if __name__ == '__main__':
    result = asyncio.run(main())
    print(result)


    
    

