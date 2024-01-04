#########################################################################
# Парсер
# Содержит определение класса Parser
# Слеповичев И.И. 20.06.2023
# Грамматика выражений:
#   module     -> definition | module definition
#   definition -> ID EQUALS expression SEMICOLON
#   expression -> FEATURES
#   expression -> ID
#   expression -> expression PLUS expression
#   expression -> expression MUL expression
#   expression -> expression POWER NUMBER
#   expression -> expression PERCENT NUMBER
#   expression -> LCBRACE expression RCBRACE
#   params     -> LPAREN param_list RPAREN
#   params     -> LPAREN RPAREN
#   param_list -> NUMBER | param_list COMMA NUMBER
#   expression -> RELU
#   expression -> SIGMOID
#   expression -> TANH
#   expression -> SOFTMAX
#   expression -> LINEAR params
#   expression -> LEAKY_RELU
#   expression -> ELU
#   expression -> SELU
#   expression -> LOG_SOFTMAX
#
#
# Токены
#   ID - идентификатор: r'[a-zA-Z_][a-zA-Z0-9_]*' ранее определнного блока, либо перечня torch.nn.functional
#   FEATURES - линейный модуль с n выходными нейронами: r'\@\d+'
#   NUMBER - число: r'\d+'
#   SEMICOLON - точка с запятой
#   EQUALS - операция присвоения результата выражения переменной: r'='
#   POWER - операция возведения в степень: r'\^'
#   LPAREN - открывающая скобка параметров (
#   RPAREN - закрывающая скобка параметров )
#   LCBRACE - открывающая фигурная скобка {
#   RCBRACE - закрывающая фигурная скобка }
#   COMMA - запятая
#   RELU - функция ReLU
#   SIGMOID - функция Sigmoid
#   TANH - функция Tanh
#   SOFTMAX - функция Softmax
#   LINEAR - функция Linear
#   LEAKY_RELU - функция LeakyReLU
#   ELU - функция ELU
#   SELU - функция SELU
#   LOG_SOFTMAX - функция LogSoftmax
#   PLUS - операция сложения: r'\+'
#   MUL - операция умножения: r'\-\>'
#   PERCENT - операция деления: r'\%'
#   COMMENT - комментарий: r'\#.*'
#
# Операции
#   Присвоение значения переменной с идентификатором x: x = @32
#   Параллельное соединение элементов с одинаковой размерностью выходного слоя: x + y
#   Композиция (последовательное соединение): x -> y
#   Копирование модуля x и последующая композиция этих n копий: x ^ n
#   Копирование модуля x и параллельное соединение n копий : x % n
#   Группировка фигурными скобками: { @4 + @4 }
#
# Примеры выражений:
#   x = @64;        # x - torch.nn.Linear 64 нейрона
#   y = x + @64;    # y - параллельно соединены x и модуль из 64 нейронов
#   z = x -> y;     # z - x последовательно соединен с y
#   w = @16 ^ 4;    # w - 4 слоя по 16 нейронов последовательно соединены
#   a = x % 2;      # a - параллельно соединены два модуля x
#   b = {{@8 -> relu + @8 -> relu} ^ 2} % 2 -> @16 -> softmax;
#########################################################################

from ply import yacc
from generator.lexer import Lexer
from generator.bricks import Activator, Adder, Composer, Connector, Identical, Linear, Multiplicator, Namer, Splitter
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import json


# TODO Добавить перехват ошибок парсинга
# TODO Создание модулей из json файлов
# TODO Восстановление описания по модулю нужно как то увязать с описаним грамматики
# TODO Добавить loss функции
# TODO Добавить torch.Tensor.view, flatten
# TODO Добавить операцию MINUS: expression -> expression MINUS ID
# TODO Создать тесты для каждого правила

# Класс парсера для создания модуля нейросети
class Parser:
    def __init__(self, lexer: Lexer = None):
        print('\nParser: инициализация ...')
        self.parser = yacc.yacc(module=self, start='module')
        if (lexer is None):
            self.lexer = Lexer()
        else:
            self.lexer = lexer
        self.reserved_funcid = self.lexer.reserved_funcid
        self.identifiers = dict()
        self.result = nn.Module()

    def __del__(self):
        self.identifiers = None
        self.lexer = None
        self.parser = None

    def from_str(self, s: str) -> dict:
        '''
        Парсинг из строки s. Вовзращает словарь с именованными модулями нейросетей.
        Одно из выражений должно иметь имя output
        Примеры:
           x = @64;        # x - torch.nn.Linear 64 нейрона
           y = x + @16;    # y - параллельно соединены x и модуль из 16 нейронов
           z = x -> y;     # z - x последовательно соединен с y
           w = @16 ^ 4;    # w - 4 слоя по 16 нейронов последовательно соединены
           a = x % 2;      # a - параллельно соединены два модуля x
           output = {{@4 -> relu + @8 -> relu} ^ 2} % 2 -> @16 -> softmax;        
        '''
        self.identifiers = dict()
        self.result = self.parser.parse(
            s, tracking=True, lexer=self.lexer.lexer)
        
        if ('output' not in self.identifiers.keys()):
            self.show_error(message=f"В модуле не определен выходной блок")
            print(list(self.identifiers.values()))
            self.identifiers['output'] = list(self.identifiers.values())[-1]
        if ('input' not in self.identifiers.keys()):
            self.show_error(message=f"В модуле не определен входной блок")
        return self.identifiers

    def from_txt(self, filename: str) -> dict:
        '''
        Парсинг из файла filename. Возвращает словарь с именованными модулями нейросетей.
        Формат файла соответствует формату строки, принимаемой на вход self.from_str
        '''
        with open(filename, 'r') as f:
            text = f.read()
        return self.from_str(text)

    def from_json(self, filename) -> dict:
        '''
        Парсинг из json-файла filename. Возвращает словарь с именованными модулями нейросетей.
        Пример:
        {
            "input": "@64",
            "c": "{ { @16->relu+@16->sigmoid }^2 } % 4 -> @16",
            "output": "input -> c -> softmax"
        }
        '''
        with open(filename, 'r') as f:
            json_dict = json.load(f)
        return self.from_dict(json_dict)

    def from_dict(self, d: dict) -> dict:
        '''
        Парсинг из словаря d. Возвращает словарь с именованными модулями нейросетей.
        Ключ - имя выражения, значение - строка выражения. Точка с запятой в выражении необязательна.
        Пример:
        {
            "input": "@64",
            "c": "{ { @16->relu+@16->sigmoid }^2 } % 4 -> @16",
            "output": "input -> c -> softmax"
        }
        '''
        s = ""
        for key in d.keys():
            s += f"{key} = {d[key].rstrip(';')};\n"
        return self.from_str(s)

    # Токены из лексера
    tokens = Lexer.tokens

    precedence = (
        ('left', 'PLUS'),
        ('left', 'MUL'),
        ('left', 'STAR'),
        ('left', 'PERCENT'),
        ('right', 'POWER'),
    )

    # Базовый элемент описания - модуль
    def p_module(self, p):
        '''module : COMMENT
                  | definition
                  | module definition'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    # Присвоение переменной значения выражения
    def p_definition(self, p):
        'definition : ID EQUALS expression SEMICOLON'
        if (p[1] in self.reserved_funcid):
            self.show_error(
                p, f"Нельзя в качестве идентификатора использовать зарезервированые функции: {p[1]}")
        self.identifiers[p[1]] = p[3]
        p[0] = p[3]

    # Число в выражении соответствует линейному модулю с соответствующим количеством нейронов
    def p_expression_number(self, p):
        'expression : FEATURES'
        n = int(p[1])
        if (n < 1 or n > 2**20):
            self.show_error(
                p, f"Количество нейронов должно быть в диапазоне от 1 до 2^20, а {n} передано")
        p[0] = Linear(n)

    # id модуля
    def p_expression_id(self, p):
        'expression : ID'
        # находится ли p[1] в списке зарезервированных слов
        if (p[1] in self.reserved_funcid.values()):
            self.show_error(
                f"Нельзя в качестве идентификатора использовать зарезервированые функции: {p[1]}")
        # наличие id в списке определенных id
        if (p[1] in self.identifiers):
            p[0] = self.identifiers[p[1]]
        else:
            self.show_error(
                p, f"Идентификатор {p[1]} не определен и не является предопределенным блоком")

    # Сложение двух выражений
    def p_expression_add(self, p):
        'expression : expression PLUS expression'
        p[0] = Adder(p[1], p[3])

    # Умножение двух выражений
    def p_expression_mul(self, p):
        'expression : expression MUL expression'
        p[0] = Composer(p[1], p[3])

    # Возведение выражения в степень NUMBER
    def p_expression_power(self, p):
        'expression : expression POWER NUMBER'
        n = int(p[3])
        if (n < 1 or n > 2**20):
            self.show_error(
                p, f"Количество должно быть в диапазоне от 1 до 2^20, а {n} передано")
        p[0] = Multiplicator(p[1], n)

    # Разделение выражения на NUMBER частей
    def p_expression_split(self, p):
        'expression : expression PERCENT NUMBER'
        n = int(p[3])
        if (n < 1 or n > 2**20):
            self.show_error(
                p, f"Количество должно быть в диапазоне от 1 до 2^20, а {n} передано")
        p[0] = Splitter(p[1], n, save_shape=True)

    def p_expression_parens(self, p):
        'expression : LCBRACE expression RCBRACE'
        p[0] = p[2]

    # Парметры функции
    def p_func_params(self, p):
        'params : LPAREN param_list RPAREN'
        p[0] = p[2]

    def p_func_params_empty(self, p):
        'params : LPAREN RPAREN'
        p[0] = []

    def p_func_param_list(self, p):
        '''param_list : NUMBER
                      | param_list COMMA NUMBER'''
        if (len(p) == 2):
            p[0] = [int(p[1])]
        else:
            p[0] = p[1] + [int(p[3])]

    # Функции
    def p_func_activator(self, p):
        '''expression : RELU
                      | SIGMOID
                      | TANH
                      | SOFTMAX
                      | LEAKY_RELU
                      | ELU
                      | SELU
                      | SOFTPLUS
                      | LOG_SOFTMAX'''
        p[0] = Activator(p[1])

    def p_func_linear(self, p):
        'expression : LINEAR params'
        params = p[2]
        if (len(params) == 0 or len(params) > 2):
            self.show_error(
                p, f"Функция {p[1]} принимает 1 параметр, а {len(params)} передано")
        p[0] = Linear(int(p[2][0]))

    def p_error(self, token):
        if token is not None:
            self.show_error(token, f"Не опознанная ошибка в '{token.value}'")
        else:
            self.show_error(token, "Неожиданный конец строки")

    def show_error(self, token=None, message=""):
        if (token is not None):
            print(f"ERROR: строке {token.lineno(1)}:\n{message}")
        else:
            print(f"ERROR: {message}")

# if __name__ == '__main__':
#     parser = NBParser()
#     if(sys.argv[1] == '' or sys.argv[2]==''):
#         print("Необходимо указать входной файл и путь для сохранения модуля:")
#         print("python nbparser.py <input_file> <output_file>")
#         exit()
#     module = parser.from_txt(sys.argv[1])
#     module.save(sys.argv[2])
#     print(module)

# paser = Parser()
# module = paser.from_str("a=((@4->elu+@8->relu)^2)%2->@16->softmax;")
# print(module)
