import unittest
import torch
import json
from generator.interpreter import Interpreter
import time

# from torchview import draw_graph
# import sys
# import torch.nn.functional as F


class TestParser(unittest.TestCase):
    def setUp(self) -> None:
        self.p = Interpreter()
        self.expressions = [
            "input = @16;\nc = { { @16->relu+@16->sigmoid}^2} % 4 -> @4;\nd = {{@64 -> relu}^4} % 2;\noutput = input -> d -> c -> @4 -> softmax;",
            "output={{@4->relu+@8->relu}^2}%2->@16->softmax->linear(5);",
            "output={{@16->relu+@16->sigmoid}^4}%8->@16;",
            "output={{@64->relu}^64}%64;",
        ]
        self.dict_expr1 = {
            "input": "@16",
            "c": "{{ @16->relu+@16->sigmoid }^2} % 4 -> @4",
            "d": "{{ @16 -> relu}^4 } % 2",
            "output": "input -> d -> c -> @4 -> softmax",
        }

        # Подготовим данные и запишем в json-файл
        self.json_filename = "src/test/test_expr.json"
        with open(self.json_filename, "w") as f:
            json.dump(self.dict_expr1, f)

        # Подготовим данные и запишем в txt-файл
        self.txt_filename = "src/test/test_expr.txt"
        with open(self.txt_filename, "w") as f:
            f.write(self.expressions[0])

        return super().setUp()

    def test_from_str(self):
        s = "output={@4->relu};"
        modules = self.p._from_str(s)
        print(f"Загрузка моделей из {s}")
        self.assertTrue("output" in modules)
        self.assertEqual(str(modules["output"]), "{linear(4)->relu}")
        self.assertEqual(str(modules["output"].left), "linear(4)")
        self.assertEqual(str(modules["output"].right), "relu")

        for expr in self.expressions:
            modules = self.p._from_str(expr)
            self.assertTrue("output" in modules)

    def test_from_json(self):
        # Теперь проверим загрузку из файла
        modules = self.p._from_json(self.json_filename)
        print(f"Загрузка моделей из {self.json_filename}")
        self.assertTrue("input" in modules)
        self.assertTrue("c" in modules)
        self.assertTrue("d" in modules)
        self.assertTrue("output" in modules)

    def test_from_txt(self):
        # Проверим загрузку из txt файла
        modules = self.p._from_txt(self.txt_filename)
        print(f"Загрузка моделей из {self.txt_filename}")
        self.assertTrue("input" in modules)
        self.assertTrue("output" in modules)


if __name__ == "__main__":
    unittest.main()


# device = "cuda"
# parser = Parser()
# s1 = "output={{@4->relu+@8->relu}^2}%2->@16->softmax->linear(5);"
# s2 = "output={{@16->relu+@16->sigmoid}^4}%8->@16;"
# s3 = "output={{@64->relu}^64}%64;"
# s4 = "output = linear(5) -> softmax;"
# s5 = "output = {@5 + @10 + @20} -> softmax;"
# start_time = time.time()
# modules = parser.from_str(s5)
# # modules = parser.from_json('nntest.json')
# model = modules['output'].to(device)
# end_time = time.time()
# print(start_time, end_time)
