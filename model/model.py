# %%
import nltk  # type: ignore
from nltk.tokenize import sent_tokenize, word_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore
from collections import Counter
import docx # type: ignore
from sklearn.decomposition import LatentDirichletAllocation # type: ignore
from sklearn.feature_extraction.text import CountVectorizer# type: ignore
from textblob import TextBlob# type: ignore
import re
nltk.download('punkt')
nltk.download('stopwords')


#* Функция для чтения документа Word
def load_text_from_word(file_path):
    doc = docx.Document(file_path)
    word_doc = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            word_doc.append(paragraph.text)
    return word_doc

#* Функция для расчета перплексии
def compute_perplexity(X, num_topics_range):
    perplexities = []
    for num_topics in num_topics_range:
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        lda.fit(X)
        perplexities.append(lda.perplexity(X))
    return perplexities

#* Функция для расчета захламленности текста
def analyze_cluster(file_path):
    #* Загрузка текста из файла Word
    word_doc = ' '.join(load_text_from_word(file_path))
    
    #* Токенизация предложений и слов
    sentences = sent_tokenize(word_doc, language='english') # Токенизация предложений
    words = word_tokenize(word_doc) # Токенизация слов

    #* Определение вводных слов
    filter_words = set([
        "actually", "basically", "just", "kind of", "like", "really", "so", "you know", "sort of", "well", "literally", "totally", "maybe", "perhaps"
        "Moreover", "Namely", "Nevertheless", "On the other hand", "Quite", "Rather", "So", "That is to say", "To sum up", "Truly", "Ultimately"
])

    #* Поиск вводных слов
    cluster_phrases = [phrase for phrase in filter_words if phrase in word_doc]
    num_cluster_phrases = len(cluster_phrases)

    #* Длина предложений
    sentence_len = [len(word_tokenize(sentence)) for sentence in sentences]
    avg_sentence_len = sum(sentence_len) / len(sentence_len) if sentences else 0

    #* Повторение слов
    stop_words = set(stopwords.words('english'))
    meaningful_words = [word for word in words if word.isalpha() and word not in stop_words]
    word_counts = Counter(meaningful_words)
    most_common = word_counts.most_common(5)

    #* Оценка захламленности
    clutter_score = 0
    max_clutter_score = 10

    #* Вводные слова: каждый найденный +2 балла
    clutter_score += num_cluster_phrases * 2

    #* Проверка на длину предложений
    if avg_sentence_len > 20:
        clutter_score += 3

    #* Повторение слов: если 5 самых частых слов встречаются более 10 раз +2 балла
    if any(count > 10 for word, count in most_common):
        clutter_score += 2

    #* Ограничение на максимальный балл
    clutter_score = min(clutter_score, max_clutter_score)
    
    if clutter_score > 0 and clutter_score < 5:
        clutter_mark = ("Текст хорошо структурирован, лишние фразы отсутствуют или встречаются редко")
    elif clutter_score > 6 and clutter_score < 10:
        clutter_mark = ("Умеренное количество лишних фраз, но текст все еще воспринимается нормально")
    else:
        clutter_mark = ("Высокая загроможденность, текст трудно читать из-за большого количества лишних слов")

    #* Формирование словаря с результатами
    return {
        "num_cluster_phrases": num_cluster_phrases,
        "avg_sentence_len": avg_sentence_len,
        "most_common_words": ', '.join([word for word, freq in most_common]),
        "clutter_score": clutter_score,
        "clutter_mark": clutter_mark
    }
    

#*Функция для оценки сентимента(настроения) текста
def analyze_sentiment(file_path):
    #* Загрузка и подготовка текста
    document = load_text_from_word(file_path)
    document_text = ' '.join(document)
    
    #* Ввожу переменную blob
    blob = TextBlob(document_text)
    sentiment = blob.sentiment
    
    #* Оценка настроения
    polarity = sentiment.polarity  # От -1 (негатив) до 1 (позитив)
    sentiment.subjectivity  # От 0 (факты) до 1 (мнение)
    
    #* Определение тональности
    if polarity > 0:
        return f"Позитивное настроение (Полярность: {polarity:.2f})"
    elif polarity < 0:
        return f"Негативное настроение (Полярность: {polarity:.2f})"
    else:
        return "Нейтральное настроение"

#* Функция для обработки текста для LDA
def process_text(file_path):
    #* Загрузка и подготовка текста
    document = load_text_from_word(file_path)
    document_text = ' '.join(document)  # Объединяю абзацы в один текст

    #* Подготовка данных
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(document)  # Преобразую текст в матрицу частот

    num_topics_range = range(1, 10)  # Пробую от 2 до 10 тем

    #* Вычисление перплексии для разных значений количества тем
    perplexities = compute_perplexity(X, num_topics_range)

    #* Определение оптимального количества тем
    optimal_num_topics = num_topics_range[perplexities.index(min(perplexities))]

    if optimal_num_topics < 3:
        optimal_num_topics = 3

    lda = LatentDirichletAllocation(n_components=optimal_num_topics, random_state=0)
    lda.fit(X)

    #* Извлечение ключевых слов для каждой темы
    terms = vectorizer.get_feature_names_out()
    unique_topics = {}
    topic_keywords = {}

    for i, topic in enumerate(lda.components_):
        top_terms_idx = topic.argsort()[-10:][::-1]
        top_terms = [terms[idx] for idx in top_terms_idx]
        hashtags = ' '.join([f"#{term}" for term in top_terms])

        if hashtags not in unique_topics:
            unique_topics[hashtags] = i
            topic_keywords[i] = top_terms
    return {
        'topic_count': optimal_num_topics,
        'topic_keywords': topic_keywords
    }

def search_words_in_text(file_path, user_query):
    document = load_text_from_word(file_path)
    document_text = ' '.join(document)
    print("Document Text:", document_text)  # Отладка
    
    #* Разделение на предложения
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', document_text)
    print("Sentences:", sentences)  # Отладка

    result = []
    
    #* Поиск ключевых слов и добавление номера предложения
    for index, sentence in enumerate(sentences, start=1):
        print("Checking Sentence:", sentence)  # Отладка
        if user_query and user_query.lower() in sentence.lower():
            result.append(f"{index}. {sentence}")
    
    return result

file_path = "C:\\Users\\User-Максим\\Desktop\\LDA.docx"
print(analyze_cluster(file_path))
print(analyze_sentiment(file_path))
process_text(file_path)


# %% [markdown]
# 


