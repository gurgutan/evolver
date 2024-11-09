
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'moduleleftPLUSleftRARROWleftPERCENTrightPOWERCOMMA COMMENT ELU EQUALS FEATURES FUNCID ID LCBRACE LEAKY_RELU LINEAR LOG_SOFTMAX LPAREN NUMBER PERCENT PLUS POWER RARROW RCBRACE RELU RPAREN SELU SEMICOLON SIGMOID SOFTMAX SOFTPLUS STAR TANH\n        module : COMMENT\n               | definition\n               | module definition\n        definition : ID EQUALS expression SEMICOLON\n        expression : expression PLUS expression\n                   | expression RARROW expression\n                   | expression POWER NUMBER\n                   | expression PERCENT NUMBER\n        expression : FEATURESexpression : IDexpression : LCBRACE expression RCBRACEparams : LPAREN param_list RPARENparams : LPAREN RPARENparam_list : NUMBER\n        | param_list COMMA NUMBERexpression : RELU\n        | SIGMOID\n        | TANH\n        | SOFTMAX\n        | LEAKY_RELU\n        | ELU\n        | SELU\n        | SOFTPLUS\n        | LOG_SOFTMAXexpression : LINEAR params'
    
_lr_action_items = {'COMMENT':([0,],[2,]),'ID':([0,1,2,3,5,6,10,21,22,23,],[4,4,-1,-2,-3,7,7,-4,7,7,]),'$end':([1,2,3,5,21,],[0,-1,-2,-3,-4,]),'EQUALS':([4,],[6,]),'FEATURES':([6,10,22,23,],[9,9,9,9,]),'LCBRACE':([6,10,22,23,],[10,10,10,10,]),'RELU':([6,10,22,23,],[11,11,11,11,]),'SIGMOID':([6,10,22,23,],[12,12,12,12,]),'TANH':([6,10,22,23,],[13,13,13,13,]),'SOFTMAX':([6,10,22,23,],[14,14,14,14,]),'LEAKY_RELU':([6,10,22,23,],[15,15,15,15,]),'ELU':([6,10,22,23,],[16,16,16,16,]),'SELU':([6,10,22,23,],[17,17,17,17,]),'SOFTPLUS':([6,10,22,23,],[18,18,18,18,]),'LOG_SOFTMAX':([6,10,22,23,],[19,19,19,19,]),'LINEAR':([6,10,22,23,],[20,20,20,20,]),'SEMICOLON':([7,8,9,11,12,13,14,15,16,17,18,19,27,29,30,31,32,33,35,37,],[-10,21,-9,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-5,-6,-7,-8,-11,-13,-12,]),'PLUS':([7,8,9,11,12,13,14,15,16,17,18,19,26,27,29,30,31,32,33,35,37,],[-10,22,-9,-16,-17,-18,-19,-20,-21,-22,-23,-24,22,-25,-5,-6,-7,-8,-11,-13,-12,]),'RARROW':([7,8,9,11,12,13,14,15,16,17,18,19,26,27,29,30,31,32,33,35,37,],[-10,23,-9,-16,-17,-18,-19,-20,-21,-22,-23,-24,23,-25,23,-6,-7,-8,-11,-13,-12,]),'POWER':([7,8,9,11,12,13,14,15,16,17,18,19,26,27,29,30,31,32,33,35,37,],[-10,24,-9,-16,-17,-18,-19,-20,-21,-22,-23,-24,24,-25,24,24,-7,-8,-11,-13,-12,]),'PERCENT':([7,8,9,11,12,13,14,15,16,17,18,19,26,27,29,30,31,32,33,35,37,],[-10,25,-9,-16,-17,-18,-19,-20,-21,-22,-23,-24,25,-25,25,25,-7,-8,-11,-13,-12,]),'RCBRACE':([7,9,11,12,13,14,15,16,17,18,19,26,27,29,30,31,32,33,35,37,],[-10,-9,-16,-17,-18,-19,-20,-21,-22,-23,-24,33,-25,-5,-6,-7,-8,-11,-13,-12,]),'LPAREN':([20,],[28,]),'NUMBER':([24,25,28,38,],[31,32,36,39,]),'RPAREN':([28,34,36,39,],[35,37,-14,-15,]),'COMMA':([34,36,39,],[38,-14,-15,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'module':([0,],[1,]),'definition':([0,1,],[3,5,]),'expression':([6,10,22,23,],[8,26,29,30,]),'params':([20,],[27,]),'param_list':([28,],[34,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> module","S'",1,None,None,None),
  ('module -> COMMENT','module',1,'p_module','grammars.py',93),
  ('module -> definition','module',1,'p_module','grammars.py',94),
  ('module -> module definition','module',2,'p_module','grammars.py',95),
  ('definition -> ID EQUALS expression SEMICOLON','definition',4,'p_definition','grammars.py',103),
  ('expression -> expression PLUS expression','expression',3,'p_expression','grammars.py',111),
  ('expression -> expression RARROW expression','expression',3,'p_expression','grammars.py',112),
  ('expression -> expression POWER NUMBER','expression',3,'p_expression','grammars.py',113),
  ('expression -> expression PERCENT NUMBER','expression',3,'p_expression','grammars.py',114),
  ('expression -> FEATURES','expression',1,'p_expression_number','grammars.py',126),
  ('expression -> ID','expression',1,'p_expression_id','grammars.py',136),
  ('expression -> LCBRACE expression RCBRACE','expression',3,'p_expression_parens','grammars.py',194),
  ('params -> LPAREN param_list RPAREN','params',3,'p_func_params','grammars.py',199),
  ('params -> LPAREN RPAREN','params',2,'p_func_params_empty','grammars.py',203),
  ('param_list -> NUMBER','param_list',1,'p_func_param_list','grammars.py',207),
  ('param_list -> param_list COMMA NUMBER','param_list',3,'p_func_param_list','grammars.py',208),
  ('expression -> RELU','expression',1,'p_func_activator','grammars.py',216),
  ('expression -> SIGMOID','expression',1,'p_func_activator','grammars.py',217),
  ('expression -> TANH','expression',1,'p_func_activator','grammars.py',218),
  ('expression -> SOFTMAX','expression',1,'p_func_activator','grammars.py',219),
  ('expression -> LEAKY_RELU','expression',1,'p_func_activator','grammars.py',220),
  ('expression -> ELU','expression',1,'p_func_activator','grammars.py',221),
  ('expression -> SELU','expression',1,'p_func_activator','grammars.py',222),
  ('expression -> SOFTPLUS','expression',1,'p_func_activator','grammars.py',223),
  ('expression -> LOG_SOFTMAX','expression',1,'p_func_activator','grammars.py',224),
  ('expression -> LINEAR params','expression',2,'p_func_linear','grammars.py',231),
]
