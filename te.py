import io # обязательные библиотеки для stremlit
import streamlit as st # # обязательные библиотеки для stremlit
from PIL import Image # библиотека для загрузки изображений
from transformers import GPT2Tokenizer, GPT2Model

#@st.cache(allow_output_mutation=True)
#def load_model():
    #return image-to-text


def load_image():
    uploaded_file = st.file_uploader(label='Загрузите пожалуйста изображение') # загрузчик файлов
    if uploaded_file is not None: # если пользователь загрузил файл
        image_data = uploaded_file.getvalue() # то мы его читаем
        st.image(image_data) # преобразуем с помощью средств stremlit
        return Image.open(io.BytesIO(image_data))# возвращаем это изображение
    else:
        return None
st.title('Классификация изображений')
img = load_image() # вызываем функцию
#mod = load_model()

result = st.button('Распознать изображение')# вставляем кнопку
st.write('**Успешно3:**')
if result: #после нажатия на которую будет запущен алгоритм...
    st.write('**Результаты распознавания:**')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
