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
#   expression -> expression RARROW expression
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
from generator.bricks import (
    Activator,
    Adder,
    Composer,
    Connector,
    Identical,
    Linear,
    Multiplicator,
    Namer,
    Splitter,
)
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import json
from loguru import logger

# TODO Восстановление описания по модулю нужно как то увязать с описаним грамматики
# TODO Добавить loss функции
# TODO Добавить torch.Tensor.view, flatten
# TODO Добавить операцию MINUS: expression -> expression MINUS ID
# TODO Создать тесты для каждого правила


# Класс парсера для создания модуля нейросети
class Parser:
    """
    A class to parse neural network module definitions from various formats.

    This class supports parsing from strings, text files, JSON files, and dictionaries.
    It constructs a representation of neural network modules based on the parsed input.
    """

    def __init__(self, lexer: Lexer = None, verbose: bool = True):
        logger.info("Инициализация парсера...")
        self.parser = yacc.yacc(module=self, start="module")
        self.lexer = lexer if lexer else Lexer()
        self.reserved_funcid = self.lexer.reserved_funcid
        self.identifiers = dict()
        self.result = nn.Module()
        self.verbose = verbose

    def from_str(self, s: str) -> dict:
        """
        Parse from a string s. Returns a dictionary with named neural network modules.
        One of the expressions must have the name output.

        Args:
            s (str): The input string to parse.

        Returns:
            dict: A dictionary of named modules.

        Examples:
            x = @64;        # x - torch.nn.Linear 64 нейрона
            output = {{@4 -> relu + @8 -> relu} ^ 2} % 2 -> @16 -> softmax;
        """
        self.identifiers = dict()
        logger.info("Парсинг скрипта...")
        self.result = self.parser.parse(s, tracking=True, lexer=self.lexer.lexer)
        logger.info(
            f"Парсинг завершен. Получены модули: {list(self.identifiers.keys())}"
        )
        if "output" not in self.identifiers:
            ids = list(self.identifiers.keys())
            self.notify(
                "В модуле не определен выходной блок output."
                f"В качестве output будет использован блок '{ids[-1]}'"
            )
            self.identifiers["output"] = list(self.identifiers.values())[-1]
        if "input" not in self.identifiers:
            logger.warning("В модуле не определен входной блок input")
        return self.identifiers

    def from_txt(self, filename: str) -> dict:
        """
        Parse a script from a file filename. Returns a dictionary with named neural network modules.
        The file format corresponds to the format of the string accepted by self.from_str.

        Args:
            filename (str): The path to the input file.

        Returns:
            dict: A dictionary of named modules.
        """
        with open(filename, "r") as f:
            text = f.read()
        return self.from_str(text)

    def from_json(self, filename: str) -> dict:
        """
        Parse from a JSON file filename. Returns a dictionary with named neural network modules.

        Args:
            filename (str): The path to the JSON file.

        Returns:
            dict: A dictionary of named modules.
        """
        with open(filename, "r") as f:
            json_dict = json.load(f)
        return self.from_dict(json_dict)

    def from_dict(self, d: dict) -> dict:
        """
        Parse from a dictionary d. Returns a dictionary with named neural network modules.
        The key is the name of the expression, and the value is the expression string.

        Args:
            d (dict): A dictionary of expressions.

        Returns:
            dict: A dictionary of named modules.
        """
        s = ""
        for key in d.keys():
            s += f"{key} = {d[key].rstrip(';')};\n"
        return self.from_str(s)

    # Токены из лексера
    tokens = Lexer.tokens

    precedence = (
        ("left", "PLUS"),
        ("left", "RARROW"),
        # ("left", "STAR"),
        ("left", "PERCENT"),
        ("right", "POWER"),
    )

    # Базовый элемент описания - модуль
    def p_module(self, p):
        """module : COMMENT
        | definition
        | module definition"""
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    # Присвоение переменной значения выражения
    def p_definition(self, p):
        "definition : ID EQUALS expression SEMICOLON"
        if p[1] in self.reserved_funcid:
            self.notify(
                p,
                f"Нельзя в качестве идентификатора использовать зарезервированые функции: {p[1]}",
            )
        self.identifiers[p[1]] = p[3]
        p[0] = p[3]

    # Число в выражении соответствует линейному модулю с соответствующим количеством нейронов
    def p_expression_number(self, p):
        "expression : FEATURES"
        try:
            n = int(p[1])
            if n < 1 or n > 2**20:
                self.notify(
                    p, f"Количество нейронов должно быть от 1 до 2^20, а не {n} "
                )
            p[0] = Linear(n)
        except Exception as e:
            self.notify(p, f"Ошибка при создании Linear: {str(e)}")

    # id модуля
    def p_expression_id(self, p):
        "expression : ID"
        if p[1] in self.reserved_funcid.values():
            self.notify(
                f"Идентификатора не должен совпадать с зарезервированой функцией: {p[1]}"
            )
        if p[1] in self.identifiers:
            p[0] = self.identifiers[p[1]]
        else:
            self.notify(
                p,
                f"Идентификатор {p[1]} не определен и не является допустимым блоком",
            )

    # Сложение двух выражений
    def p_expression_add(self, p):
        "expression : expression PLUS expression"
        try:
            p[0] = Adder(p[1], p[3])
        except Exception as e:
            self.notify(p, f"Ошибка при создании Adder: {str(e)}")

    # Умножение двух выражений
    def p_expression_mul(self, p):
        "expression : expression RARROW expression"
        try:
            p[0] = Composer(p[1], p[3])
        except Exception as e:
            self.notify(p, f"Ошибка при создании Composer: {str(e)}")

    # Возведение выражения в степень NUMBER
    def p_expression_power(self, p):
        "expression : expression POWER NUMBER"
        try:
            n = int(p[3])
            if n < 1 or n > 2**20:
                self.notify(
                    p,
                    f"Число n в выражении 'expr ^ n' должно быть от 1 до 2**20, а не {n}",
                )
            p[0] = Multiplicator(p[1], n)
        except Exception as e:
            self.notify(p, f"Ошибка при создании Multiplicator: {str(e)}")

    # Разделение выражения на NUMBER частей
    def p_expression_split(self, p):
        "expression : expression PERCENT NUMBER"
        try:
            n = int(p[3])
            if n < 1 or n > 2**20:
                self.notify(
                    p,
                    f"Число n в выражении 'expr % n' должно быть от 1 до 2^20, а не {n}",
                )
            p[0] = Splitter(p[1], n, save_shape=True)
        except Exception as e:
            self.notify(p, f"Ошибка при создании Splitter: {str(e)}")

    def p_expression_parens(self, p):
        "expression : LCBRACE expression RCBRACE"
        p[0] = p[2]

    # Парметры функции
    def p_func_params(self, p):
        "params : LPAREN param_list RPAREN"
        p[0] = p[2]

    def p_func_params_empty(self, p):
        "params : LPAREN RPAREN"
        p[0] = []

    def p_func_param_list(self, p):
        """param_list : NUMBER
        | param_list COMMA NUMBER"""
        if len(p) == 2:
            p[0] = [int(p[1])]
        else:
            p[0] = p[1] + [int(p[3])]

    # Функции
    def p_func_activator(self, p):
        """expression : RELU
        | SIGMOID
        | TANH
        | SOFTMAX
        | LEAKY_RELU
        | ELU
        | SELU
        | SOFTPLUS
        | LOG_SOFTMAX"""
        try:
            p[0] = Activator(p[1])
        except Exception as e:
            self.notify(p, f"Ошибка при создании Activator: {str(e)}")

    def p_func_linear(self, p):
        "expression : LINEAR params"
        try:
            params = p[2]
            if len(params) == 0 or len(params) > 2:
                self.notify(
                    p, f"Функция {p[1]} принимает 1 параметр, а {len(params)} передано"
                )
            p[0] = Linear(int(p[2][0]))
        except Exception as e:
            self.notify(p, f"Ошибка при создании Linear: {str(e)}")

    def p_error(self, token):
        if token is not None:
            self.notify(token, f"Не опознанная ошибка в '{token.value}'")
        else:
            self.notify(token, "Неожиданный конец строки")

    def notify(self, token=None, message=""):
        if token is not None:
            logger.error(f"Строка {token.lineno(1)}:\n{message}")
        else:
            logger.error(f"{message}")
