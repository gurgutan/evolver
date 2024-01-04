import streamlit as st
import torch
from torchview import draw_graph
from  generator.parser import Parser
import generator.bricks as bricks
import sys
import time
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

help_text = """
Синтаксис скрипта для создания модели:
   <id1> = <expr1>;
   <id2> = <expr2>;
   ...
   output = <exprN>;   
   где <id1>, <idN> - идентификаторы модели, а <exprN> - выражение, описывающее модель
   
   В выражении можно использовать символ '@' для создания линейного элемента,
   бинарные операторы '->' и '+' для последовательного и параллельного соединения,
   а также скобки '{','}' для группировки. Выражение вседа заканчивается символом ';'.
   В выражении должен присутствовать идентификатор 'output'/
   
Операции:
   =   присвоение,   
   @   создание линейного элемента (тоже, что и linear),   
   ->  создание последовательно соединеных элементов,   
   +   создание параллельно соединенных элементов (размерность должна совпадать),   
   {}  скобки группировки выражений,
   ()  скобки группировки параметров функций.
   
Поэлементные функции:
   rely, tanh, sigmoid, elu, selu

Примеры:
   x = @32;            # Присвоение значения переменной с идентификатором x
   output = @8 + @8;   # Параллельное соединение линейных элементов из 8 нейронов
   output = @8 -> @8;  # Композиция (последовательное соединение)
   output = @16 ^ 4;   # Копирование линейного элемента @16 и последующая композиция этих 4 копий
   output = x % 16;    # Копирование модуля x и параллельное соединение 16 копий
   output = {@4 + @4}; # Группировка фигурными скобками

Пример скрипта из нескольких выражений:
   x = @64;            # x - nn.Linear 64 нейрона
   y = x + @64;        # y - параллельно соединены x и модуль из 64 нейронов
   z = x -> y;         # z - x последовательно соединен с y
   w = @8 ^ 4;         # w - 4 слоя по 16 нейронов последовательно соединены
   a = x % 2;          # a - параллельно соединены два модуля x
   output = z -> w -> a -> {{@8 -> relu + @8 -> relu} ^ 2} % 2 -> @16 -> softmax;
   
*В данной демострации размерность входного тензора (1, 8)
"""


# Задаём размеры изображения и шрифт
IMAGE_SIZE = (500, 500)


def generate_model(expr: str) -> dict:
    """Функция возвращает словарь с модулями, сгенерированными по входной строке expr"""
    print(text)
    start_time = time.time()
    # создаем парсер
    parser = Parser() 
    # создаем модели из строки expr
    modules = parser.from_str(expr)
    # # Создаем модули их json-файла
    # modules = parser.from_json('nntest.json')
    # Выбираем модуль, который записан в переменную output
    end_time = time.time()
    print(f"Время генерации модели: {end_time-start_time}")
    return modules


def module_params(model: torch.nn.Module) -> int:
    params_count = sum(p.numel() for p in model.parameters())
    return params_count


def generate_image(model: torch.nn.Module):
    # Создаём изображение и объект для рисования
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # Тестовой тензор
    shape = (1, 8)
    x = torch.rand(shape).float().to(device)
    y = model.to(device)(x)
    pic_path = './pic/'
    image_name = 'test'
    model_graph = draw_graph(
        model,
        input_data=x,
        # input_size=shape,
        graph_name='test',
        graph_dir='LR',
        depth=8,
        hide_inner_tensors=True,
        hide_module_functions=True,
        save_graph=True,
        expand_nested=True,
        show_shapes=True,
        filename=image_name,
        directory=pic_path
    )
    image = Image.open(pic_path + image_name + '.png')
    return image

# Задаём заголовок приложения
st.title("Нейросеть по описанию")
st.subheader("Демонстрация генерации модели нейросети по описанию на специальном языке")
# st.subheader("Демонстрация языка генерации НН моделей:")


# left_column, right_column = st.columns(2)
# Создаём поле для ввода текста
text = st.text_area("Введите выражение:", value='output = @4 -> {@256 + @256} % 4 -> @64 -> softmax;')

if st.checkbox("Подсказка синтаксиса"):
    st.text(help_text)

# Если текст введён, генерируем изображение и выводим его
if text:
    st.write("Модель")
    modules = generate_model(text)
    model = modules['output']
    image = generate_image(model)
    params_count = module_params(model)
    st.image(image)
    st.write(f"Модель: {modules}\n\nПараметров: {params_count}")
    
