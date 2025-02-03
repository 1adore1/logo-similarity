## Task & Solution

Задание: 
Предложи систему, которая позволит по кропу области с произвольным логотипом на изображении (берется из видео, куда нативно встроена реклама) отвечать на вопрос: "Является ли он логотипом искомой организации?"

В качестве логотипов искомой организации дается несколько образцов.
Заранее неизвестно, что именно эта организация будет искаться. Логотипы могут быть текстовыми и нетекстовыми.

На изображении в реальности данные логотипы могут иметь самый различный масштаб, могут иметь отличные характеристики по яркости, контрастности, могут быть повернуты, несколько искажены, иметь небольшие отличия в дизайне.

Если получится, продемонстрируй работу системы на примерах.

Из open source датасетов есть такой вариант https://paperswithcode.com/dataset/logodet-3k, если будет полезно.

В поле ответа дай описание системы и, если есть прототип, приложи ссылку на код с примерами в репозитории на
GitHub, GitLab, Bitbucket или любой другой с открытым доступом.


Ответ: 
Для данной задачи подойдет создание эмбеддингов референсных и тестовых изображений и их сравнение по какой-либо метрике, обычно это косинусное сходство или евклидово расстояние. Для построения эмбеддингов можно воспользоваться классическими алгоритмами (SIFT, HOG), либо CNN в качестве feature extractor, но обычно нейросетевые решения показывают лучший результат.

Было решено использовать ResNet-50 для построения эмбеддингов логотипов и косинусное сходство в качестве метрики похожести логотипов. Модель обучалась на открытом датасете logodet-3k. Логотип каждой компании был вырезан в соответствии с его bounding box в .xml файле. Функция потерь - tripletloss с косинусным сходством, то есть на каждой итерации берется 2 элемента из одного класса (якорь и положительный объект), 1 из другого (отрицательный объект). Максимизируется сходство между якорем и положительным объектом, минимизируется между якорем и отрицательным объектом. 

Также для удобства был реализован веб интерфейс (с помощью streamlit) для загрузки референсных и тестовых изображений и вывода результата сравнения эмбеддингов.

Веб интерфейс, для тестирования проекта: https://logo-similarity.streamlit.app/

# Logo Similarity

This project was a test assignment for VK. It demonstrates a system that, given a cropped region of an image (e.g., a logo extracted from a video with embedded advertising), answers the question: 

**"Is this the logo of the target organization?"**

The system compares the provided test logo against several reference logo samples of the organization. It is designed to handle logos with varying scales, brightness, contrast, rotations, and minor design variations. The underlying idea is to use a deep learning model to generate embeddings for each logo and then compute the similarity (via cosine similarity) to decide if the test logo matches one of the provided samples.

Try it: [logo-similarity.streamlit.app](https://logo-similarity.streamlit.app/)

## How It Works

1. **Upload Reference Logos:** Users can upload multiple images representing the target organization's logos.
2. **Upload Test Logo:** Users upload a cropped image containing the test logo.
3. **Comparison:** The system processes each image through a deep learning model, computes embeddings, and calculates cosine similarity between the test logo and each reference.
4. **Result Display:** The system displays the similarity score and a match/no-match result based on a predefined threshold.

## Getting Started

### Requirements

- Python 3.x
- [Streamlit](https://streamlit.io/)
- PyTorch
- PIL

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/1adore1/logo-similarity.git
   cd logo-similarity
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   
### Running the Application

Launch the app using Streamlit:
```
streamlit run src/main.py
```

Then, open the provided URL in your browser to interact with the system.


## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Model**: `ResNet50`
