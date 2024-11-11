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

from pathlib import Path
from typing import Dict, Union
import torch.nn as nn
import json

from ply import yacc
from generator.errors import ParserError
from generator.grammars import AnnetGrammar
from generator.lexer import Tokenizer
from loguru import logger

# TODO Восстановление описания по модулю нужно как то увязать с описаним грамматики
# TODO Добавить loss функции
# TODO Добавить torch.Tensor.view, flatten
# TODO Добавить операцию MINUS: expression -> expression MINUS ID
# TODO Создать тесты для каждого правила


# Класс парсера для создания модуля нейросети
class Parser(AnnetGrammar):
    """
    A class to parse neural network module definitions from various formats.

    This class supports parsing from strings, text files, JSON files, and dictionaries.
    It constructs a representation of neural network modules based on the parsed input.
    """

    OUTPUT_ID = "output"
    INPUT_ID = "input"

    def __init__(self, lexer: Tokenizer = None, verbose: bool = True):
        super().__init__(lexer or Tokenizer(), verbose)
        logger.info("Инициализация парсера...")
        self.parser = yacc.yacc(module=self, start="module")
        self.result = None
        self.verbose = verbose

    def from_str(self, s: str) -> Dict[str, nn.Module]:
        """
        Parse a string and return a dictionary of named neural network modules.

        Args:
            s (str): The input string to parse.

        Returns:
            Dict[str, nn.Module]: A dictionary of named modules.

        Raises:
            ParserError: If there's an error during parsing.

        Examples:
            x = @64;        # x - torch.nn.Linear shape (64,)
            output = {{@4 -> relu + @8 -> relu} ^ 2} % 2 -> @16 -> softmax;
        """
        self.identifiers.clear()
        try:
            self.result = self.parser.parse(
                s, tracking=True, lexer=self.tokenizer.lexer
            )
            self._check_output()
            return self.identifiers
        except ParserError as e:
            logger.error(str(e))
            raise

    def from_txt(self, filename: str) -> dict:
        """
        Parse a script from a file filename. Returns a dictionary with named neural network modules.
        The file format corresponds to the format of the string accepted by self.from_str.

        Args:
            filename (str): The path to the input file.

        Returns:
            dict: A dictionary of named modules.
        """
        text = Path(filename).read_text()
        return self.from_str(text)

    def from_dict(self, d: dict) -> dict:
        """
        Parse from a dictionary d. Returns a dictionary with named neural network modules.
        The key is the name of the expression, and the value is the expression string.

        Args:
            d (dict): A dictionary of expressions.

        Returns:
            dict: A dictionary of named modules.
        """
        s = ";\n".join(f"{key} = {value.rstrip(';')}" for key, value in d.items())
        return self.from_str(f"{s};")

    def from_json(self, filename: str) -> dict:
        """
        Parse from a JSON file filename. Returns a dictionary with named neural network modules.

        Args:
            filename (str): The path to the JSON file.

        Returns:
            dict: A dictionary of named modules.
        """
        json_dict = json.loads(Path(filename).read_text())
        return self.from_dict(json_dict)

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
        validated_input = self._validate_input(input_data)
        return self.from_str(validated_input)

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
            return self.from_dict(input_data)
        elif isinstance(input_data, Path):
            if input_data.suffix == ".json":
                return self.from_json(input_data)
            elif input_data.suffix == ".txt":
                return self._read_file(input_data)
            else:
                raise ParserError(f"Unsupported file type: {input_data.suffix}")
        else:
            raise ParserError(f"Unsupported input type: {type(input_data)}")

    def _check_output(self):
        """
        Check if 'output' and 'input' blocks are defined in the parsed module.
        If 'output' is not defined, use the last defined block as output.
        """
        if not self.identifiers:
            raise ParserError("No modules were defined during parsing.")

        if Parser.OUTPUT_ID not in self.identifiers:
            last_id = list(self.identifiers.keys())[-1]
            logger.warning(f"Output block not defined. Using '{last_id}' as output.")
            self.identifiers[Parser.OUTPUT_ID] = self.identifiers[last_id]

        if not isinstance(self.identifiers[Parser.OUTPUT_ID], nn.Module):
            raise ParserError(
                "The 'output' block is not a valid neural network module, but\n"
                f"{type(self.identifiers[Parser.OUTPUT_ID])}"
            )

        if Parser.INPUT_ID not in self.identifiers:
            logger.warning("Input block not defined in the module.")
