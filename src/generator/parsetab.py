# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = "3.10"

_lr_method = "LALR"

_lr_signature = "scriptleftPLUSleftRARROWleftPERCENTrightPOWERCOMMA COMMENT ELU EQUALS ID LCBRACE LEAKY_RELU LINEAR LOG_SOFTMAX LPAREN NUMBER PERCENT PLUS POWER RARROW RCBRACE RELU RPAREN SELU SEMICOLON SHAPE SIGMOID SOFTMAX SOFTPLUS STAR STRING TANHscript : assigments\n        assigments : COMMENT\n                   | assigment\n                   | assigments assigment\n        assigment : ID EQUALS expression SEMICOLON\n        expression : expression PLUS expression\n                   | expression RARROW expression\n                   | expression POWER NUMBER\n                   | expression PERCENT NUMBER\n        expression : LCBRACE expression RCBRACEexpression : SHAPEexpression : STRINGexpression : IDparams : LPAREN param_list RPARENparams : LPAREN RPAREN\n        param_list : parameter\n                   | param_list COMMA parameter\n        \n        parameter : NUMBER\n                  | STRING\n        \n        expression : RELU\n                   | SIGMOID\n                   | TANH\n                   | SOFTMAX\n                   | LEAKY_RELU\n                   | ELU\n                   | SELU\n                   | SOFTPLUS\n                   | LOG_SOFTMAX\n        expression : LINEAR params"

_lr_action_items = {
    "COMMENT": (
        [
            0,
        ],
        [
            3,
        ],
    ),
    "ID": (
        [
            0,
            2,
            3,
            4,
            6,
            7,
            10,
            23,
            24,
            25,
        ],
        [
            5,
            5,
            -2,
            -3,
            -4,
            8,
            8,
            -5,
            8,
            8,
        ],
    ),
    "$end": (
        [
            1,
            2,
            3,
            4,
            6,
            23,
        ],
        [
            0,
            -1,
            -2,
            -3,
            -4,
            -5,
        ],
    ),
    "EQUALS": (
        [
            5,
        ],
        [
            7,
        ],
    ),
    "LCBRACE": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            10,
            10,
            10,
            10,
        ],
    ),
    "SHAPE": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            11,
            11,
            11,
            11,
        ],
    ),
    "STRING": (
        [
            7,
            10,
            24,
            25,
            30,
            42,
        ],
        [
            12,
            12,
            12,
            12,
            40,
            40,
        ],
    ),
    "RELU": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            13,
            13,
            13,
            13,
        ],
    ),
    "SIGMOID": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            14,
            14,
            14,
            14,
        ],
    ),
    "TANH": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            15,
            15,
            15,
            15,
        ],
    ),
    "SOFTMAX": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            16,
            16,
            16,
            16,
        ],
    ),
    "LEAKY_RELU": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            17,
            17,
            17,
            17,
        ],
    ),
    "ELU": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            18,
            18,
            18,
            18,
        ],
    ),
    "SELU": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            19,
            19,
            19,
            19,
        ],
    ),
    "SOFTPLUS": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            20,
            20,
            20,
            20,
        ],
    ),
    "LOG_SOFTMAX": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            21,
            21,
            21,
            21,
        ],
    ),
    "LINEAR": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            22,
            22,
            22,
            22,
        ],
    ),
    "SEMICOLON": (
        [
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            29,
            31,
            32,
            33,
            34,
            35,
            37,
            41,
        ],
        [
            -13,
            23,
            -11,
            -12,
            -20,
            -21,
            -22,
            -23,
            -24,
            -25,
            -26,
            -27,
            -28,
            -29,
            -6,
            -7,
            -8,
            -9,
            -10,
            -15,
            -14,
        ],
    ),
    "PLUS": (
        [
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            28,
            29,
            31,
            32,
            33,
            34,
            35,
            37,
            41,
        ],
        [
            -13,
            24,
            -11,
            -12,
            -20,
            -21,
            -22,
            -23,
            -24,
            -25,
            -26,
            -27,
            -28,
            24,
            -29,
            -6,
            -7,
            -8,
            -9,
            -10,
            -15,
            -14,
        ],
    ),
    "RARROW": (
        [
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            28,
            29,
            31,
            32,
            33,
            34,
            35,
            37,
            41,
        ],
        [
            -13,
            25,
            -11,
            -12,
            -20,
            -21,
            -22,
            -23,
            -24,
            -25,
            -26,
            -27,
            -28,
            25,
            -29,
            25,
            -7,
            -8,
            -9,
            -10,
            -15,
            -14,
        ],
    ),
    "POWER": (
        [
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            28,
            29,
            31,
            32,
            33,
            34,
            35,
            37,
            41,
        ],
        [
            -13,
            26,
            -11,
            -12,
            -20,
            -21,
            -22,
            -23,
            -24,
            -25,
            -26,
            -27,
            -28,
            26,
            -29,
            26,
            26,
            -8,
            -9,
            -10,
            -15,
            -14,
        ],
    ),
    "PERCENT": (
        [
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            28,
            29,
            31,
            32,
            33,
            34,
            35,
            37,
            41,
        ],
        [
            -13,
            27,
            -11,
            -12,
            -20,
            -21,
            -22,
            -23,
            -24,
            -25,
            -26,
            -27,
            -28,
            27,
            -29,
            27,
            27,
            -8,
            -9,
            -10,
            -15,
            -14,
        ],
    ),
    "RCBRACE": (
        [
            8,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            28,
            29,
            31,
            32,
            33,
            34,
            35,
            37,
            41,
        ],
        [
            -13,
            -11,
            -12,
            -20,
            -21,
            -22,
            -23,
            -24,
            -25,
            -26,
            -27,
            -28,
            35,
            -29,
            -6,
            -7,
            -8,
            -9,
            -10,
            -15,
            -14,
        ],
    ),
    "LPAREN": (
        [
            22,
        ],
        [
            30,
        ],
    ),
    "NUMBER": (
        [
            26,
            27,
            30,
            42,
        ],
        [
            33,
            34,
            39,
            39,
        ],
    ),
    "RPAREN": (
        [
            30,
            36,
            38,
            39,
            40,
            43,
        ],
        [
            37,
            41,
            -16,
            -18,
            -19,
            -17,
        ],
    ),
    "COMMA": (
        [
            36,
            38,
            39,
            40,
            43,
        ],
        [
            42,
            -16,
            -18,
            -19,
            -17,
        ],
    ),
}

_lr_action = {}
for _k, _v in _lr_action_items.items():
    for _x, _y in zip(_v[0], _v[1]):
        if not _x in _lr_action:
            _lr_action[_x] = {}
        _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {
    "script": (
        [
            0,
        ],
        [
            1,
        ],
    ),
    "assigments": (
        [
            0,
        ],
        [
            2,
        ],
    ),
    "assigment": (
        [
            0,
            2,
        ],
        [
            4,
            6,
        ],
    ),
    "expression": (
        [
            7,
            10,
            24,
            25,
        ],
        [
            9,
            28,
            31,
            32,
        ],
    ),
    "params": (
        [
            22,
        ],
        [
            29,
        ],
    ),
    "param_list": (
        [
            30,
        ],
        [
            36,
        ],
    ),
    "parameter": (
        [
            30,
            42,
        ],
        [
            38,
            43,
        ],
    ),
}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
    for _x, _y in zip(_v[0], _v[1]):
        if not _x in _lr_goto:
            _lr_goto[_x] = {}
        _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
    ("S' -> script", "S'", 1, None, None, None),
    ("script -> assigments", "script", 1, "p_script", "grammars.py", 110),
    ("assigments -> COMMENT", "assigments", 1, "p_assigments", "grammars.py", 116),
    ("assigments -> assigment", "assigments", 1, "p_assigments", "grammars.py", 117),
    (
        "assigments -> assigments assigment",
        "assigments",
        2,
        "p_assigments",
        "grammars.py",
        118,
    ),
    (
        "assigment -> ID EQUALS expression SEMICOLON",
        "assigment",
        4,
        "p_assigment",
        "grammars.py",
        128,
    ),
    (
        "expression -> expression PLUS expression",
        "expression",
        3,
        "p_expression",
        "grammars.py",
        136,
    ),
    (
        "expression -> expression RARROW expression",
        "expression",
        3,
        "p_expression",
        "grammars.py",
        137,
    ),
    (
        "expression -> expression POWER NUMBER",
        "expression",
        3,
        "p_expression",
        "grammars.py",
        138,
    ),
    (
        "expression -> expression PERCENT NUMBER",
        "expression",
        3,
        "p_expression",
        "grammars.py",
        139,
    ),
    (
        "expression -> LCBRACE expression RCBRACE",
        "expression",
        3,
        "p_expression_braces",
        "grammars.py",
        151,
    ),
    ("expression -> SHAPE", "expression", 1, "p_expression_shape", "grammars.py", 155),
    (
        "expression -> STRING",
        "expression",
        1,
        "p_expression_string",
        "grammars.py",
        165,
    ),
    ("expression -> ID", "expression", 1, "p_expression_id", "grammars.py", 169),
    (
        "params -> LPAREN param_list RPAREN",
        "params",
        3,
        "p_func_params",
        "grammars.py",
        178,
    ),
    ("params -> LPAREN RPAREN", "params", 2, "p_func_params_empty", "grammars.py", 182),
    (
        "param_list -> parameter",
        "param_list",
        1,
        "p_func_param_list",
        "grammars.py",
        187,
    ),
    (
        "param_list -> param_list COMMA parameter",
        "param_list",
        3,
        "p_func_param_list",
        "grammars.py",
        188,
    ),
    ("parameter -> NUMBER", "parameter", 1, "p_func_parameter", "grammars.py", 197),
    ("parameter -> STRING", "parameter", 1, "p_func_parameter", "grammars.py", 198),
    ("expression -> RELU", "expression", 1, "p_func_activator", "grammars.py", 209),
    ("expression -> SIGMOID", "expression", 1, "p_func_activator", "grammars.py", 210),
    ("expression -> TANH", "expression", 1, "p_func_activator", "grammars.py", 211),
    ("expression -> SOFTMAX", "expression", 1, "p_func_activator", "grammars.py", 212),
    (
        "expression -> LEAKY_RELU",
        "expression",
        1,
        "p_func_activator",
        "grammars.py",
        213,
    ),
    ("expression -> ELU", "expression", 1, "p_func_activator", "grammars.py", 214),
    ("expression -> SELU", "expression", 1, "p_func_activator", "grammars.py", 215),
    ("expression -> SOFTPLUS", "expression", 1, "p_func_activator", "grammars.py", 216),
    (
        "expression -> LOG_SOFTMAX",
        "expression",
        1,
        "p_func_activator",
        "grammars.py",
        217,
    ),
    (
        "expression -> LINEAR params",
        "expression",
        2,
        "p_func_linear",
        "grammars.py",
        225,
    ),
]
