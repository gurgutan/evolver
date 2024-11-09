class ParserError(Exception):
    def __init__(self, message: str, token=None):
        self.message = message
        self.token = token
        super().__init__(self.message)

    def __str__(self):
        if self.token:
            return f"Line {self.token.lineno}: {self.message}"
        return self.message
