"""
Sample Python file for testing CodeParser.
"""


class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        """Return a greeting message."""
        return f"Hello, {self.name}!"


def sample_function(x: int, y: int) -> int:
    """Add two numbers together."""
    return x + y


def another_function():
    """Another sample function."""
    pass
