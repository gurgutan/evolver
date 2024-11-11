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

import re
from typing import Dict
from loguru import logger
from generator.bricks import Activator, Adder, Composer, Linear, Multiplicator, Splitter
from generator.lexer import Tokenizer


class AnnetGrammar:
    tokens = Tokenizer.tokens

    precedence = (
        ("left", "PLUS"),
        ("left", "RARROW"),
        ("left", "PERCENT"),
        ("right", "POWER"),
    )

    def __init__(self, tokenizer: Tokenizer, verbose: bool = True):
        self.tokenizer = tokenizer or Tokenizer()
        self.identifiers = {}

    def p_module(self, p):
        """
        module : COMMENT
               | definition
               | module definition
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    @property
    def reserved_ids(self) -> Dict[str, str]:
        return self.tokenizer.reserved_ids

    def p_definition(self, p):
        "definition : ID EQUALS expression SEMICOLON"
        if p[1] in self.reserved_ids:
            self.notify(p, f"ID can't be reserved word: {p[1]}")
        self.identifiers[p[1]] = p[3]
        p[0] = p[3]

    def p_expression(self, p):
        """
        expression : expression PLUS expression
                   | expression RARROW expression
                   | expression POWER NUMBER
                   | expression PERCENT NUMBER
        """
        if re.fullmatch(Tokenizer.t_PLUS, p[2]):
            p[0] = Adder(p[1], p[3])
        elif re.fullmatch(Tokenizer.t_RARROW, p[2]):
            p[0] = Composer(p[1], p[3])
        elif re.fullmatch(Tokenizer.t_POWER, p[2]):
            p[0] = Multiplicator(p[1], int(p[3]))
        elif re.fullmatch(Tokenizer.t_PERCENT, p[2]):
            p[0] = Splitter(p[1], int(p[3]), save_shape=True)

    def p_expression_number(self, p):
        "expression : FEATURES"
        try:
            n = int(p[1])
            if not (0 < n <= 2**10):
                self.notify(p, f"Numbers must be in range [1, 2**10], а не {n} ")
            p[0] = Linear(n)
        except Exception as e:
            self.notify(p, f"Error Linear creation: {str(e)}")

    def p_expression_id(self, p):
        "expression : ID"
        if p[1] in self.reserved_ids.values():
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
        """
        expression : RELU
                   | SIGMOID
                   | TANH
                   | SOFTMAX
                   | LEAKY_RELU
                   | ELU
                   | SELU
                   | SOFTPLUS
                   | LOG_SOFTMAX
        """
        try:
            p[0] = Activator(p[1])
        except Exception as e:
            self.notify(p, f"Error in creation Activator: {str(e)}")

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
            self.notify(token, f"Unknown error '{token.value}'")
        else:
            self.notify(token, "Неожиданный конец строки")

    def notify(self, token=None, message=""):
        if token is not None:
            logger.error(f"Line {token.lineno(1)}:\n{message}")
        else:
            logger.error(f"{message}")