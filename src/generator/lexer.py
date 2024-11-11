################################################################
# Лексер
# Содержит определение класса NBLexer
# Слеповичев И.И. 20.11.2024
################################################################

import secrets
import string
import ply.lex as lex


class Tokenizer:
    def __init__(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def __del__(self):
        pass

    # Зарезервиврованные слова
    reserved_ids = {
        "relu": "RELU",
        "sigmoid": "SIGMOID",
        "tanh": "TANH",
        "softmax": "SOFTMAX",
        "leaky_relu": "LEAKY_RELU",
        "elu": "ELU",
        "selu": "SELU",
        "softplus": "SOFTPLUS",
        "log_softmax": "LOG_SOFTMAX",
        "linear": "LINEAR",
        # TODO определения других функций стандартных блоков НС
        #'conv1d': 'CONV1D',
        #'conv2d': 'CONV2D',
        #'conv3d': 'CONV3D',
        #'maxpool1d': 'MAXPOOL1D',
        #'maxpool2d': 'MAXPOOL2,D',
        #'maxpool3d': 'MAXPOOL3D',
        #'avgpool1d': 'AVGPOOL1D',
        #'avgpool2d': 'AVGPOOL2D',
        #'avgpool3d': 'AVGPOOL3D',
        #'adaptiveavgpool1': 'ADAPTIVEAVGPOOL1D',
        #'adaptiveavgpool2': 'ADAPTIVEAVGPOOL2D',
        #'adaptiveavgpool3': 'ADAPTIVEAVGPOOL3D',
        #'batchnorm1d': 'BATCHNORM1D',
        #'batchnorm2d': 'BATCHNORM2D',
        #'batchnorm3d': 'BATCHNORM3D',
        #'dropout': 'DROPOUT',
        #'flatten': 'FLATTEN',
        #'embedding': 'EMBEDDING',
        #'lstm': 'LSTM',
        #'gru': 'GRU',
        #'rnn': 'RNN'
    }

    # list of TOKENS
    tokens = [
        "SHAPE",
        "COMMA",
        "NUMBER",
        "ID",
        "PLUS",
        "STAR",
        "RARROW",
        "PERCENT",
        "LPAREN",
        "RPAREN",
        "LCBRACE",
        "RCBRACE",
        "EQUALS",
        "POWER",
        "SEMICOLON",
        "COMMENT",
    ] + list(reserved_ids.values())

    t_COMMA = r"\,"
    t_PLUS = r"\+"
    t_STAR = r"\*"
    t_RARROW = r"\-\>"
    t_PERCENT = r"\%"
    t_LPAREN = r"\("
    t_RPAREN = r"\)"
    t_LCBRACE = r"\{"
    t_RCBRACE = r"\}"
    t_EQUALS = r"="
    t_SEMICOLON = r"\;"
    t_POWER = r"\^"
    t_ignore = " \t"

    def t_newline(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)

    def t_SHAPE(self, t):
        r"\@\d+"
        t.value = int(t.value[1:])
        return t

    def t_NUMBER(self, t):
        r"\d+"
        t.value = int(t.value)
        return t

    def t_ID(self, t):
        r"[a-zA-Z_][a-zA-Z0-9_]*"
        t.type = self.reserved_ids.get(t.value, "ID")
        return t

    # def t_FUNCID(self, t):
    #     r'\@(relu|sigmoid|tanh|softmax|leaky_relu|elu|selu|softplus|log_softmax|linear)'
    #     t.value = t.value[1:]  # убираем символ '@' в начале
    #     return t

    def t_error(self, t):
        print(f"Invalid character '{t.value[0]}'")
        t.lexer.skip(1)

    def t_COMMENT(self, t):
        r"\#.*"
        pass

    def t_nl(self, t):
        r"(\n|\r|\r\n)|\s|\t"


def generate_identifier(length=16):
    valid_chars = string.ascii_lowercase + string.digits
    random_string = "id_" + "".join(secrets.choice(valid_chars) for _ in range(length))
    return random_string
