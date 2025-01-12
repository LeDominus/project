import re
import language_tool_python
import pdfplumber
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BartModel, BartTokenizer
import asyncio
from tqdm.asyncio import tqdm_asyncio
import textstat
import re


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

'''
СЕКЦИЯ АНАЛИЗА ТЕКСТА
'''

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

#* Функция для извлечения структуры из текста
def extract_structure(text, section_patterns):
    structure = []
    for pattern in section_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            print(f"Найдено совпадение по шаблону {pattern}: {matches[0]}")
            structure.append(matches[0])
    return structure

#* Функция для анализа структуры текста
'''
original_text - тот, который сравниваю
reference_text - эталон(шаблон)
'''

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

    if not original_structure or not reference_structure:
        print("Ошибка: не удалось извлечь разделы из одного или обоих текстов.")
        return 0.0
    
    if original_structure == reference_structure:
        return 1.0
    else:
        matching_sections = set(original_structure) & set(reference_structure)
        structure_similarity =  len(matching_sections) / len(reference_structure) if len(reference_structure) > 0 else 0.0
    
        if structure_similarity > 0.85:
            structure_interpret = 'Структура текста соответствует стандартам'
        elif 0.5 < structure_similarity <= 0.85:
            structure_interpret = 'Структура текста требует доработки'
        else:
            structure_interpret = 'Структура текста не соответствует стандартам'
            
        return structure_similarity, structure_interpret

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
        return 0.0, 'Недостаточно данных для анализа когерентности'
    
    embeddings = await asyncio.gather(*[get_embeddings_async(section) for section in sections])
    
    coherence_score = []
    for i in range(1, len(embeddings)):
        similarity = torch.nn.functional.cosine_similarity(embeddings[i-1], embeddings[i], dim=-1)
        avg_similarity = similarity.mean().item()
        coherence_score.append(avg_similarity)
    
    if not coherence_score:
        print('Ошибка: не удалось рассчитать когерентность, отсутствуют данные для сравнения.')
        return 0.0, 'Ошибка в данных для расчета когерентности'

    average_coherence = sum(coherence_score) / len(coherence_score) if len(coherence_score) > 0 else 0.0
    print(f'Средний коэффициент когерентности: {average_coherence}')
    
    if average_coherence > 0.85:
        coherence_interpret = 'Текст имеет связную структуру'
    elif 0.5 < average_coherence <= 0.85:
        coherence_interpret = 'Текст имеет проблемы с логикой'
    else:
        coherence_interpret = 'Текст логически не связан'
    
    return average_coherence, coherence_interpret

#* Асинхронная функция для подсчёта индекса Флеша
async def flesch_reading_ease(text):
    return await asyncio.to_thread(textstat.flesch_reading_ease, text)

#* Асинхронная функция для подсчёта индекса Ганнинга
async def gunning_fog_index(text):
    return await asyncio.to_thread(textstat.gunning_fog, text)

#* Асинхронная функция для подсчёта числа предложений, слов и слогов
async def count_sentences_words_syllables(text):
    sentences = re.split(r'[.!?]+', text)
    words = re.findall(r'\w+', text)

    syllable_tasks = [asyncio.to_thread(textstat.syllable_count, word) for word in words]
    
    syllables = sum(await asyncio.gather(*syllable_tasks))
    
    return len(sentences), len(words), syllables

#* Асинхронная функция для оценки уровня образования на основе индекса читаемости
async def readability_level(flesch_score):
    if flesch_score >= 90:
        return "Очень легко (для младших классов)"
    elif flesch_score >= 60:
        return "Легко (для средней школы)"
    elif flesch_score >= 30:
        return "Трудно (для колледжа)"
    else:
        return "Очень трудно (для специалистов)"

#* Основная асинхронная функция для анализа текста
async def analyze_readability(text):
    flesch_score = await flesch_reading_ease(text)
    gunning_score = await gunning_fog_index(text)
    
    sentences, words, syllables = await count_sentences_words_syllables(text)
    
    readability = await readability_level(flesch_score)
    
    analysis_result = {
        "Индекс Флеша": flesch_score,
        "Индекс Ганнинга": gunning_score,
        "Количество предложений": sentences,
        "Количество слов": words,
        "Слоги": syllables,
        "Сложность текста": readability
    }
    
    return analysis_result

async def show_loading(computation_task):
    progress_bar = tqdm_asyncio(total=1, desc='Обработка результатов', unit='task')
    result = await computation_task
    progress_bar.update(1)
    progress_bar.close()
    return result

'''
СЕКЦИЯ ФУНКЦИИ ДЛЯ ЗАПУСКА 
'''

#* Основная функция для запуска
async def main():
    file_path_original = 'C:\\Programming\\Python\\Python developing\\python_basics\\Project_BERT\\PDF_учебники\\Гладун И. В_Статистика.pdf'
    file_path_reference = 'C:\\Programming\\Python\\Python developing\\python_basics\\Project_BERT\\PDF_учебники\\УчП_Стат_методы_анализа_Шорохова_и_др.pdf'
    
    original_text = convert_text_from_pdf(file_path_original)
    reference_text = convert_text_from_pdf(file_path_reference)

    print('Проверка стиля...')
    style_original_text = classify_style(original_text)
       
    coherence_task = asyncio.create_task(analyze_coherence(reference_text))
    coherence_score = await show_loading(coherence_task)
    
    structure_task = asyncio.create_task(asyncio.to_thread(analyze_structure, original_text, reference_text))
    structure_score = await show_loading(structure_task)
    
    read_score = await analyze_readability(original_text)
       
    result = {
        "style_result": style_original_text,
        "coherence_result": coherence_score,
        "structure_result": structure_score,
        "read_result" : read_score
    }
    
    return result

# СТАРТУЕМ
if __name__ == '__main__':
    
    result = asyncio.run(main())
    print(result)


    
    

