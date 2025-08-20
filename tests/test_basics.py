"""
Basic tests to validate core Python concepts for the workshop.
These tests help Java/C# developers verify their understanding of Python fundamentals.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_python_version():
    """Ensure we're using Python 3.11+"""
    assert sys.version_info >= (3, 11), "This workshop requires Python 3.11 or higher"

def test_basic_data_types():
    """Test understanding of Python's dynamic typing vs Java/C# static typing"""
    # Python allows reassignment to different types (unlike Java/C#)
    var = "Hello"  # String
    assert isinstance(var, str)
    
    var = 42  # Integer
    assert isinstance(var, int)
    
    var = 3.14  # Float
    assert isinstance(var, float)
    
    var = True  # Boolean
    assert isinstance(var, bool)

def test_list_comprehensions():
    """Test Pythonic list operations vs traditional loops"""
    numbers = [1, 2, 3, 4, 5]
    
    # Pythonic way (equivalent to Java streams or C# LINQ)
    squares = [x**2 for x in numbers]
    evens = [x for x in numbers if x % 2 == 0]
    
    assert squares == [1, 4, 9, 16, 25]
    assert evens == [2, 4]

def test_dict_operations():
    """Test Python dictionary operations vs Java HashMap/C# Dictionary"""
    person = {"name": "Alice", "age": 30, "city": "New York"}
    
    # Safe access with default (like Java's getOrDefault)
    country = person.get("country", "Unknown")
    assert country == "Unknown"
    
    # Dictionary comprehension
    upper_keys = {k.upper(): v for k, v in person.items()}
    assert "NAME" in upper_keys
    assert upper_keys["NAME"] == "Alice"

def test_function_features():
    """Test Python's flexible function features"""
    def greet(name, greeting="Hello", punctuation="!"):
        return f"{greeting}, {name}{punctuation}"
    
    # Default parameters
    assert greet("Alice") == "Hello, Alice!"
    
    # Keyword arguments (order doesn't matter)
    assert greet("Bob", punctuation=".", greeting="Hi") == "Hi, Bob."
