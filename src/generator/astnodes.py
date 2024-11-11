import json
from typing import Dict


class ASTNode:
    """Базовый класс для узлов AST."""

    def serialize(self) -> dict:
        """Сериализует узел в словарь."""
        raise NotImplementedError("Must be implemented in subclasses.")


class AssigmentNode(ASTNode):
    """Узел присваивания значения переменной."""

    def __init__(self, name: str, value: ASTNode):
        self.name = name
        self.value = value

    def serialize(self) -> dict:
        """Сериализует узел присваивания в словарь."""
        return {
            "type": "assignment",
            "name": self.name,
            "value": self.value.serialize(),
        }


class OperationNode(ASTNode):
    """Узел операции, представляющий бинарные операции."""

    def __init__(self, left: ASTNode, right: ASTNode, operator: str):
        self.left = left
        self.right = right
        self.operator = operator

    def serialize(self) -> dict:
        """Сериализует узел операции в словарь."""
        return {
            "type": "operation",
            "operator": self.operator,
            "left": self.left.serialize(),
            "right": self.right.serialize(),
        }


class ModuleNode(ASTNode):
    """Узел модуля, представляющий модули из bricks."""

    def __init__(self, module_type: str, params: list = None):
        self.module_type = module_type
        self.params = params or []

    def serialize(self) -> dict:
        """Сериализует узел модуля в словарь."""
        return {
            "type": "module",
            "module_type": self.module_type,
            "params": self.params,
        }


class IdentifierNode(ASTNode):
    """Узел идентификатора, представляющий переменные."""

    def __init__(self, name: str):
        self.name = name

    def serialize(self) -> dict:
        """Сериализует узел идентификатора в словарь."""
        return {
            "type": "identifier",
            "name": self.name,
        }


class NumberNode(ASTNode):
    """Узел числа, представляющий числовые значения."""

    def __init__(self, value: int):
        self.value = value

    def serialize(self) -> dict:
        """Сериализует узел числа в словарь."""
        return {
            "type": "number",
            "value": self.value,
        }


class ScriptNode(ASTNode):
    """Узел скрипта, представляющий скрипт в нативном формате."""

    def __init__(self, assigments: dict = None):
        super().__init__()
        self.assigments: Dict[str, AssigmentNode] = assigments or {}

    def serialize(self) -> dict:
        """Сериализует узел скрипта в словарь."""
        return {
            "type": "script",
            "name": self.name,
            "modules": {m.name: m.value.serialize() for m in self.assigments.values()},
        }


# Пример использования
if __name__ == "__main__":
    # Создание простого AST
    ast = OperationNode(
        left=ModuleNode("Linear", [64]), right=IdentifierNode("x"), operator="plus"
    )

    # Сериализация AST
    serialized_ast = ast.serialize()
    print(json.dumps(serialized_ast, indent=2))
