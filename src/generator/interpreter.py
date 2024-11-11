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

import copy
from pathlib import Path
import time
from typing import Dict, Union
import torch.nn as nn
import json

from ply import yacc
from generator.errors import ParserError
from generator.grammars import AnnetParser
from generator.lexer import Tokenizer
from loguru import logger
from generator.bricks import (
    Activator,
    Sum,
    Composition,
    Linear,
    Multiplicator,
    Splitter,
)
from generator.astnodes import (
    ASTNode,
    AssigmentNode,
    OperationNode,
    ModuleNode,
    IdentifierNode,
    NumberNode,
    ScriptNode,
)

# TODO Восстановление описания по модулю нужно как то увязать с описаним грамматики
# TODO Добавить loss функции
# TODO Добавить torch.Tensor.view, flatten
# TODO Добавить операцию MINUS: expression -> expression MINUS ID
# TODO Создать тесты для каждого правила


# Класс парсера для создания модуля нейросети
class Interpreter(AnnetParser):
    """
    A class to parse neural network module definitions from various formats.

    This class supports parsing from strings, text files, JSON files, and dictionaries.
    It constructs a representation of neural network modules based on the parsed input.
    """

    OUTPUT_ID = "output"
    INPUT_ID = "input"

    def __init__(self, lexer: Tokenizer = None, verbose: bool = True):
        super().__init__(lexer or Tokenizer(), verbose)
        self.parser = yacc.yacc(module=self, start="script")
        self.ast = None
        self.verbose = verbose
        self.modules = dict()
        self.statistics = dict()

    def parse(
        self, input_data: Union[str, Dict[str, str], Path]
    ) -> Dict[str, nn.Module]:
        """
        Parse input data and return a dictionary of named neural network modules.

        Args:
            input_data: The input data to parse. Can be a string, dictionary, or file path.

        Returns:
            Dict[str, nn.Module]: A dictionary of named modules.

        Raises:
            ParserError: If there's an error during parsing.
        """
        start_time = time.time()
        validated_input = self._validate_input(input_data)
        result = self._from_str(validated_input)
        end_time = time.time()

        self.statistics["compile_time"] = end_time - start_time
        logger.info(f"Time to generate models: {end_time-start_time}")
        return result

    def _from_str(self, s: str) -> Dict[str, nn.Module]:
        """
        Parse a string and return a dictionary of named neural network modules.

        Args:
            s (str): The input string to parse.

        Returns:
            Dict[str, nn.Module]: A dictionary of named modules.

        Examples:
            x = @64;        # x - torch.nn.Linear shape (64,)
            output = {{@4 -> relu + @8 -> relu} ^ 2} % 2 -> @16 -> softmax;
        """
        self.statistics.clear()
        self.modules.clear()
        self.ast = None
        try:
            self.ast = self.parser.parse(s, tracking=True, lexer=self.tokenizer.lexer)
            self.script = self.ast_to_script(self.ast_root)
            return copy.deepcopy(self.modules)
        except ParserError as e:
            logger.error(str(e))
            raise

    def _from_txt(self, filename: str) -> dict:
        text = Path(filename).read_text()
        return self._from_str(text)

    def _from_dict(self, d: dict) -> dict:
        s = ";\n".join(f"{key} = {value.rstrip(';')}" for key, value in d.items())
        return self._from_str(f"{s};")

    def _from_json(self, filename: str) -> dict:
        json_dict = json.loads(Path(filename).read_text())
        return self._from_dict(json_dict)

    def _validate_input(self, input_data: Union[str, Dict[str, str], Path]) -> str:
        """
        Validate and prepare input data for parsing.

        Args:
            input_data: The input data to validate. Can be a string, dictionary, or file path.

        Returns:
            str: The validated and prepared input string.

        Raises:
            ParserError: If the input data is invalid or cannot be read.
        """
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return self._from_dict(input_data)
        elif isinstance(input_data, Path):
            if input_data.suffix == ".json":
                return self._from_json(input_data)
            elif input_data.suffix == ".annet":
                return self._read_file(input_data)
            else:
                raise ParserError(f"Unsupported file type: {input_data.suffix}")
        else:
            raise ParserError(f"Unsupported input type: {type(input_data)}")

    def _read_file(self, filename: str) -> str:
        return Path(filename).read_text()

    def ast_to_script(self, node: ASTNode):
        """
        Преобразует узел AST в конечный модуль.

        Args:
            node (ASTNode): Узел AST.

        Returns:
            nn.Module: Конечный модуль.
        """
        if isinstance(node, ScriptNode):
            for assigment in node.assigments:
                self.ast_to_script(assigment)
            return self.modules
        elif isinstance(node, ModuleNode):
            if node.module_type == "Linear":
                return Linear(*node.params)
            elif node.module_type == "Activator":
                return Activator(node.params[0])
            # Добавить другие модули по мере необходимости
            raise ValueError(f"Unknown module type: {node.module_type}")

        elif isinstance(node, OperationNode):
            left_module = self.ast_to_script(node.left)
            right_module = self.ast_to_script(node.right)

            if node.operator == "plus":
                return Sum(left_module, right_module)
            elif node.operator == "arrow":
                return Composition(left_module, right_module)
            elif node.operator == "power":
                return Multiplicator(left_module, node.right.value)
            elif node.operator == "percent":
                return Splitter(left_module, node.right.value)

        elif isinstance(node, AssigmentNode):
            module = self.ast_to_script(node.value)
            self.modules[node.name] = module
            return module

        elif isinstance(node, NumberNode):
            return Linear(node.value)

        elif isinstance(node, IdentifierNode):
            if node.name in self.modules:
                return self.modules[node.name]
            elif node.name in self.assigments:
                raise ValueError(f"Identifier {node.name} used before definition.")
            raise ValueError(f"Identifier {node.name} not found.")

        raise ValueError("Invalid AST node.")
