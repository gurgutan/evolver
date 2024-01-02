# Базовые модули для построения модели:
# Activator, Adder, Composer, Connector, Identical, Linear, Multiplicator, Namer, Splitter


import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class Brick(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def decompose(self):
        pass


class ActivationFunction(Brick):
    @abstractmethod
    def activate(self):
        pass


class ReluFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "relu"


class SigmoidFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "sigmoid"


class TanhFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "tanh"


class LeakyReluFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "leaky_relu"


class EluFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "elu"


class SeluFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.selu(x)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self)
        right = copy.deepcopy(self).to(self)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "selu"


class SoftplusFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "softplus"


class SoftmaxFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=1)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "softmax"


class LogSoftmaxFunction(ActivationFunction):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=1)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает две функции того же типа"""
        return self.split()

    def __name__(self):
        return "log_softmax"

# TODO Переделать Activator так, чтобы использовались классы функций выше


class Activator(nn.Module):
    def __init__(self, func_name: str, device: str = "cpu"):
        super(Activator, self).__init__()
        self.func_name = func_name
        functions = {
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh,
            "leaky_relu": F.leaky_relu,
            "elu": F.elu,
            "selu": F.selu,
            "softplus": F.softplus,
        }
        functions_dim = {
            "softmax": F.softmax,
            "log_softmax": F.log_softmax
        }
        if (self.func_name in functions):
            self.f = functions[self.func_name]
        elif (self.func_name in functions_dim):
            self.f = functions_dim[self.func_name]
        else:
            raise ValueError(f"Unknown function {self.func_name}")
        self.has_dim = func_name in functions_dim
        self.device = device
        self.to(self.device)

    # Допустимые функции:
    # relu, sigmoid, tanh, softmax, leaky_relu, elu, selu, softplus, log_softmax, linear
    def forward(self, x) -> torch.Tensor:
        if (self.has_dim):
            return self.f(x, dim=1)
        else:
            return self.f(x)

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def decompose(self):
        """Возвращает декомпозицию модуля self на два модуля left, right"""
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def __name__(self):
        return self.func_name

    def __repr__(self) -> str:
        return self.func_name

    def expr_str(self, expand=True):
        return f"{self.func_name}"


class Adder(nn.Module):
    def __init__(self, left: nn.Module, right: nn.Module, device: str = "cpu"):
        """
        Initializes a new instance of the class. Output shapes of left and right must be equal
        
        Args:
            left (nn.Module): The left module.
            right (nn.Module): The right module.
            device (str, optional): The device to use. Defaults to "cpu".
        """
        super().__init__()
        # assert left.shape == right.shape, f"Shapes must be equal, but got {left.shape} and {right.shape}"
        self.left = left
        self.right = right
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.left(x), self.right(x)]
        return torch.stack(y, dim=1).sum(dim=1)*0.5

    def split(self):
        """
        Возвращает две копии модуля self: left, right так, что
        self(x) равно (left(x)+right(x))*0.5 или self(x) == AddBrick(left, right)(x)
        """
        left, right = self.left.split(), self.right.split()
        return left, right

    def decompose(self):
        """
        Возвращает self.left.decompose(), self.right.decompose()
        """
        return self.left.decompose(), self.right.decompose()

    def __name__(self):
        return "adder"

    def __repr__(self):
        return "{" + f"{self.left.__repr__()}+{self.right.__repr__()}" + "}"

    def expr_str(self, expand=True):
        return "{" + f"{self.left.expr_str(expand)}+{self.right.expr_str(expand)}" + "}"


class Composer(nn.Module):
    def __init__(self, left: nn.Module, right: nn.Module, device: str = "cpu"):
        super().__init__()
        self.left = left
        self.right = right
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(x))

    def split(self):
        """Возвращает left.split(), right.split()"""
        return self.left.split(), self.right.split()

    def decompose(self):
        """Возвращает left.decompose(), right.decompose()"""
        return self.left.decompose(), self.right.decompose()

    def __name__(self):
        return "composer"

    def __repr__(self):
        return "{" + f"{self.left.__repr__()}->{self.right.__repr__()}" + "}"

    def expr_str(self, expand=True):
        return "{" + f"{self.left.expr_str(expand)}->{self.right.expr_str(expand)}" + "}"


class Connector(nn.Module):
    def __init__(self, left: nn.Module, right: nn.Module, device: str = "cpu"):
        super().__init__()
        self.left = left
        self.right = right
        self.cat = torch.cat
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cat([self.left(x), self.right(x)], dim=1)

    def split(self):
        """Возвращает left.split(), right.split()"""
        return self.left.split(), self.right.split()

    def decompose(self):
        """Возвращает left.decompose(), right.decompose()"""
        return self.left.decompose(), self.right.decompose()

    def __name__(self):
        return "connector"

    def __repr__(self):
        return "{" + f"{self.left.__repr__()}+{self.right.__repr__()}" + "}"

    def expr_str(self, expand=True):
        return "{" + f"{self.left.expr_str(expand)}+{self.right.expr_str(expand)}" + "}"


class Identical(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.to(self.device)

    def __name__(self):
        return "id"

    def __repr__(self):
        return f"id()"

    def forward(self, x) -> torch.Tensor:
        return x

    def split(self):
        """Возвращает две копии модуля self: left, right"""
        left = Identical(self.device)
        right = Identical(self.device)
        return left, right

    def decompose(self):
        """
        Возвращает декомпозицию модуля self на два модуля left, right так, что
        self(x) равно right(left(x)) или, в других обозначениях, self(x) == MulModule(left, right)(x)
        """
        left = Identical(self.device)
        right = Identical(self.device)
        return left, right


class Linear(nn.Module):
    def __init__(self, out_features: int, weights: torch.Tensor = None, biases: torch.Tensor = None, device: str = "cpu"):
        super().__init__()
        if (weights is None or biases is None):
            self.linear = nn.LazyLinear(out_features)
        else:
            self.linear = nn.Linear(weights.shape[1], weights.shape[0])
            self.linear.weight.data = weights
            self.linear.bias.data = biases

        self.device = device
        self.to(self.device)

    def __name__(self):
        return "linear"

    def __repr__(self):
        return f"linear({self.linear.out_features})"

    def decompose(self):
        """
        Возвращает декомпозицию модуля self на два модуля left, right так, что
        self(x) равно right(left(x)) или, в других обозначениях, self(x) равно MulModule(left, right)(x)
        Для weight используется QR-декомпозиция.
        Для bias используется правило left.linear.bias = [0,...,0], right.linear.bias = self.linear.bias
        """
        with torch.no_grad():
            # Применяем QR-разложение к матрице self.linear.weight
            W2, W1 = torch.linalg.qr(self.linear.weight)
            b1 = torch.zeros(W1.shape[0])
            b2 = self.linear.bias
            left = Linear(W1.shape[0], W1, b1, self.device)
            right = Linear(W2.shape[0], W2, b2, self.device)
        return left, right

    def split(self):
        """
        Возвращает две копии модуля self: left, right так, что
        self(x) равно (left(x)+right(x))*0.5 или self(x) == AddBrick(left, right)(x)
        """
        left = copy.deepcopy(self).to(self.device)
        right = copy.deepcopy(self).to(self.device)
        return left, right

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)

    def expr_str(self, expand=True):
        return f"linear({self.linear.out_features})"


class Multiplicator(nn.Module):
    def __init__(self, module: nn.Module, p: int, device: str = "cpu"):
        super().__init__()
        assert p > 0, "p must be positive"
        self.layers = nn.ModuleList([copy.deepcopy(module) for _ in range(p)])
        self.pow = p
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def split(self):
        """Возвращает left.split(), right.split()"""
        return self.left.split(), self.right.split()

    def decompose(self):
        """Возвращает left.decompose(), right.decompose()"""
        return self.left.decompose(), self.right.decompose()

    def __name__(self):
        return "multiplicator"

    def __repr__(self):
        return f"{self.layers[0].__repr__()}^{self.pow}"

    def expr_str(self, expand=True):
        return f"{self.layers[0].expr_str(expand)}^{self.pow}"


class Splitter(nn.Module):
    """
    Соединяет параллельно split копий модуля brick    
    """

    def __init__(self, brick: nn.Module, split: int, save_shape: bool = True, device: str = "cpu"):
        super().__init__()
        '''
        Соединяет параллельно split копий модуля module
        save_shape - если True, то применяется сумма тензоров по dim=1, иначе конкатенация
        '''
        super().__init__()
        assert split > 0, "split must be positive"
        self.layers = nn.ModuleList([copy.deepcopy(brick)
                                    for _ in range(split)])
        self.split = split
        self.save_shape = save_shape
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [layer(x) for layer in self.layers]
        if (self.save_shape):
            stack = torch.stack(y, dim=1)
            return stack.sum(dim=1)/self.split
        else:
            return torch.cat(y, dim=1)

    def __name__(self):
        return "splitter"

    def __repr__(self):
        return f"{self.layers[0].__repr__()}%{self.split}"

    def expr_str(self, expand=True):
        return f"{self.layers[0].expr_str(expand)}%{self.split}"


class Namer(nn.Module):
    """Применяется для присвоения имени модулю"""

    def __init__(self, module: nn.Module, name: str):
        super().__init__()
        self.module = module
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def expr_str(self, expand=True):
        if (expand):
            return f"{self.name}={self.module.expr_str(expand)};"
        else:
            return self.name
