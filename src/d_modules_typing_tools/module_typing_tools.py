"""
Modules, Typing, and Tools Workshop - Practice Exercises

Complete the following tasks to practice module organization, type hints, and development tools.
Apply the principles from the markdown file to create well-structured, type-safe Python code.
"""

from typing import Any, Protocol
from abc import ABC, abstractmethod

# =============================================================================
# TASK 1: Basic Module Organization
# =============================================================================

"""
TASK 1: Create a Math Utilities Module

Create a module with mathematical utility functions.

Requirements:
- Function name: calculate_circle_area
- Parameter: radius (float)
- Return: float (area of circle)
- Function name: calculate_rectangle_area  
- Parameters: length (float), width (float)
- Return: float (area of rectangle)
- Function name: calculate_distance
- Parameters: point1 (tuple of two floats), point2 (tuple of two floats)
- Return: float (Euclidean distance)
- Include module-level constants: PI, GOLDEN_RATIO
- Use proper type hints for all functions
- Include comprehensive docstrings with examples

Example usage:
calculate_circle_area(5.0) -> 78.54
calculate_rectangle_area(4.0, 6.0) -> 24.0
calculate_distance((0, 0), (3, 4)) -> 5.0
"""

# Module constants
PI: float = 3.14159265359
GOLDEN_RATIO: float = 1.618033988749

def calculate_circle_area(radius: float) -> float:
    """Your solution here"""
    pass

def calculate_rectangle_area(length: float, width: float) -> float:
    """Your solution here"""
    pass

def calculate_distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """Your solution here"""
    pass

# =============================================================================
# TASK 2: Advanced Type Hints with Generics
# =============================================================================

"""
TASK 2: Create a Generic Repository Pattern

Create a generic repository class that can store and retrieve any type of data.

Requirements:
- Class name: Repository
- Use Generic[T] to make it work with any type
- Methods: save(id: str, item: T) -> None
- Methods: get(id: str) -> T | None
- Methods: get_all() -> list[T]
- Methods: delete(id: str) -> bool
- Use proper type hints with TypeVar
- Include comprehensive docstrings
- Handle edge cases (duplicate IDs, missing items)

Example usage:
user_repo = Repository[User]()
user_repo.save("1", User("Alice"))
user = user_repo.get("1")  # Returns User or None
"""

from typing import TypeVar, Generic

T = TypeVar('T')

class Repository(Generic[T]):
    """Your solution here"""
    pass

# =============================================================================
# TASK 3: Protocol-Based Design
# =============================================================================

"""
TASK 3: Create Drawable Protocol and Implementations

Create a protocol for drawable objects and implement concrete classes.

Requirements:
- Protocol name: Drawable
- Method: draw() -> str
- Method: get_area() -> float
- Create classes: Circle, Rectangle, Triangle
- All classes should implement the Drawable protocol
- Function name: render_shapes
- Parameter: shapes (list of Drawable objects)
- Return: list[str] (list of rendered shapes)
- Use proper type hints with Protocol

Example usage:
circle = Circle(5.0)
rectangle = Rectangle(4.0, 6.0)
shapes = [circle, rectangle]
rendered = render_shapes(shapes) -> ["Circle with radius 5.0", "Rectangle 4.0x6.0"]
"""

class Drawable(Protocol):
    """Your protocol here"""
    pass

class Circle:
    """Your solution here"""
    pass

class Rectangle:
    """Your solution here"""
    pass

class Triangle:
    """Your solution here"""
    pass

def render_shapes(shapes: list[Drawable]) -> list[str]:
    """Your solution here"""
    pass

# =============================================================================
# TASK 4: Literal Types and Overloads
# =============================================================================

"""
TASK 4: Create a Configuration Manager with Literal Types

Create a configuration manager that handles different types of configuration values.

Requirements:
- Use Literal types for configuration keys
- ConfigKey = Literal["database_url", "debug_mode", "api_timeout"]
- Function name: get_config
- Use @overload to provide different return types based on key
- database_url -> str, debug_mode -> bool, api_timeout -> int
- Function name: set_config  
- Parameters: key (ConfigKey), value (str | bool | int)
- Include validation for value types
- Use proper type hints and overloads

Example usage:
set_config("database_url", "postgresql://localhost/db")
url = get_config("database_url")  # Type checker knows this is str
set_config("debug_mode", True)
debug = get_config("debug_mode")  # Type checker knows this is bool
"""

from typing import Literal, overload

ConfigKey = Literal["database_url", "debug_mode", "api_timeout"]

# Global configuration storage
_config: dict[str, str | bool | int] = {}

@overload
def get_config(key: Literal["database_url"]) -> str: ...

@overload  
def get_config(key: Literal["debug_mode"]) -> bool: ...

@overload
def get_config(key: Literal["api_timeout"]) -> int: ...

def get_config(key: ConfigKey) -> str | bool | int:
    """Your solution here"""
    pass

def set_config(key: ConfigKey, value: str | bool | int) -> None:
    """Your solution here"""
    pass

# =============================================================================
# TASK 5: Data Classes with Validation
# =============================================================================

"""
TASK 5: Create User Data Class with Validation

Create a user data class with comprehensive validation and type safety.

Requirements:
- Use @dataclass decorator
- Class name: User
- Fields: id (int), name (str), email (str), age (int), tags (list[str])
- Use field() with default_factory for mutable defaults
- Implement __post_init__ for validation
- Method: add_tag(tag: str) -> None
- Method: remove_tag(tag: str) -> bool
- Method: to_dict() -> dict[str, Any]
- Validate email contains '@'
- Validate age is between 0 and 150
- Include proper type hints

Example usage:
user = User(1, "Alice", "alice@example.com", 25)
user.add_tag("admin")
data = user.to_dict()
"""

from dataclasses import dataclass, field

@dataclass
class User:
    """Your solution here"""
    pass

# =============================================================================
# TASK 6: Module Package Structure
# =============================================================================

"""
TASK 6: Create Package Structure Simulation

Simulate a package structure with proper imports and __all__ definitions.

Requirements:
- Create classes that simulate a models package
- Class name: BaseModel (abstract base class)
- Class name: Product (inherits from BaseModel)
- Class name: Order (inherits from BaseModel)
- Function name: get_all_models
- Return: list[type[BaseModel]] (list of all model classes)
- Use ABC and abstractmethod
- Define __all__ list for public API
- Use proper type hints for class hierarchies

Example usage:
models = get_all_models() -> [Product, Order]
product = Product("Laptop", 999.99)
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Your solution here"""
    pass

class Product(BaseModel):
    """Your solution here"""
    pass

class Order(BaseModel):
    """Your solution here"""
    pass

def get_all_models() -> list[type[BaseModel]]:
    """Your solution here"""
    pass

# Public API - what gets imported with "from module import *"
__all__ = ["BaseModel", "Product", "Order", "get_all_models"]

# =============================================================================
# TASK 7: Error Handling with Custom Types
# =============================================================================

"""
TASK 7: Create Result Type for Error Handling

Create a Result type that represents either success or failure, similar to Rust's Result type.

Requirements:
- Use Union types and dataclasses
- Class name: Success (generic dataclass)
- Class name: Error (dataclass with message)
- Type alias: Result[T] = Success[T] | Error
- Function name: divide_safely
- Parameters: a (float), b (float)
- Return: Result[float]
- Function name: process_results
- Parameter: results (list[Result[float]])
- Return: list[float] (only successful values)
- Include proper type hints and error messages

Example usage:
result = divide_safely(10, 2) -> Success(5.0)
result = divide_safely(10, 0) -> Error("Division by zero")
values = process_results([Success(1.0), Error("fail"), Success(2.0)]) -> [1.0, 2.0]
"""

from dataclasses import dataclass
from typing import Union

@dataclass
class Success(Generic[T]):
    """Your solution here"""
    pass

@dataclass
class Error:
    """Your solution here"""
    pass

Result = Success[T] | Error

def divide_safely(a: float, b: float) -> Result[float]:
    """Your solution here"""
    pass

def process_results(results: list[Result[float]]) -> list[float]:
    """Your solution here"""
    pass

# =============================================================================
# TASK 8: Type Guards and Runtime Type Checking
# =============================================================================

"""
TASK 8: Create Type Guards for Runtime Validation

Create type guard functions that validate types at runtime.

Requirements:
- Function name: is_string_list
- Parameter: obj (Any)
- Return: TypeGuard[list[str]]
- Function name: is_user_dict
- Parameter: obj (Any) 
- Return: TypeGuard[dict[str, Any]] (validates User-like dict)
- Function name: process_mixed_data
- Parameter: data (list[Any])
- Return: tuple[list[str], list[dict[str, Any]]] (separated valid data)
- Use TypeGuard for type narrowing
- Include runtime validation logic

Example usage:
mixed = ["hello", {"name": "Alice", "age": 25}, "world", {"invalid": "data"}]
strings, users = process_mixed_data(mixed) -> (["hello", "world"], [{"name": "Alice", "age": 25}])
"""

from typing import TypeGuard

def is_string_list(obj: Any) -> TypeGuard[list[str]]:
    """Your solution here"""
    pass

def is_user_dict(obj: Any) -> TypeGuard[dict[str, Any]]:
    """Your solution here"""
    pass

def process_mixed_data(data: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Your solution here"""
    pass

# =============================================================================
# TASK 9: Comprehensive Package with Testing Support
# =============================================================================

"""
TASK 9: Create a Statistics Package

Create a comprehensive statistics package with proper module organization.

Requirements:
- Class name: Dataset
- Methods: add_value(value: float) -> None
- Methods: mean() -> float | None
- Methods: median() -> float | None  
- Methods: std_dev() -> float | None
- Function name: create_sample_dataset
- Parameter: size (int), seed (int | None = None)
- Return: Dataset (with random values)
- Include comprehensive error handling
- Use proper type hints throughout
- Include validation for edge cases (empty dataset)

Example usage:
dataset = Dataset()
dataset.add_value(1.0)
dataset.add_value(2.0) 
mean_val = dataset.mean() -> 1.5
sample = create_sample_dataset(100) -> Dataset with 100 random values
"""

import random
import math
from typing import Optional

class Dataset:
    """Your solution here"""
    pass

def create_sample_dataset(size: int, seed: int | None = None) -> Dataset:
    """Your solution here"""
    pass

# =============================================================================
# BONUS TASK: Advanced Type System Features
# =============================================================================

"""
BONUS TASK: Create a Type-Safe Configuration System

Create an advanced configuration system using multiple type system features.

Requirements:
- Use Literal, Generic, Protocol, overload, and dataclasses
- Create ConfigValue protocol with get() -> T method
- Create StringConfig, IntConfig, BoolConfig classes implementing ConfigValue
- Create ConfigManager class that stores different config types
- Method: register_config(key: str, config: ConfigValue[T]) -> None
- Method: get_config(key: str) -> Any (with proper type narrowing)
- Use TypeVar for generic config values
- Include comprehensive validation and error handling

Example usage:
manager = ConfigManager()
manager.register_config("db_url", StringConfig("postgresql://..."))
manager.register_config("timeout", IntConfig(30))
url = manager.get_config("db_url")  # Should be typed as Any but validate to str
"""

class ConfigValue(Protocol[T]):
    """Your protocol here"""
    pass

class StringConfig:
    """Your solution here"""
    pass

class IntConfig:
    """Your solution here"""
    pass

class BoolConfig:
    """Your solution here"""
    pass

class ConfigManager:
    """Your solution here"""
    pass

# =============================================================================
# TEST FUNCTIONS - Run these to check your solutions
# =============================================================================

def test_all_tasks():
    """Test all implemented functions"""
    print("Testing Modules, Typing, and Tools implementations...\n")
    
    # Test Task 1
    print("Task 1 - Math Utilities:")
    try:
        area = calculate_circle_area(5.0)
        rect_area = calculate_rectangle_area(4.0, 6.0)
        distance = calculate_distance((0, 0), (3, 4))
        print(f"Circle area (r=5): {area}")
        print(f"Rectangle area (4x6): {rect_area}")
        print(f"Distance (0,0) to (3,4): {distance}")
        print("✓ Task 1 working\n")
    except Exception as e:
        print(f"✗ Task 1 error: {e}\n")
    
    # Test Task 2
    print("Task 2 - Generic Repository:")
    try:
        repo = Repository[str]()
        repo.save("1", "Hello")
        repo.save("2", "World")
        item = repo.get("1")
        all_items = repo.get_all()
        deleted = repo.delete("1")
        print(f"Retrieved item: {item}")
        print(f"All items: {all_items}")
        print(f"Deleted successfully: {deleted}")
        print("✓ Task 2 working\n")
    except Exception as e:
        print(f"✗ Task 2 error: {e}\n")
    
    # Test Task 3
    print("Task 3 - Protocol Design:")
    try:
        circle = Circle(5.0)
        rectangle = Rectangle(4.0, 6.0)
        shapes = [circle, rectangle]
        rendered = render_shapes(shapes)
        print(f"Rendered shapes: {rendered}")
        print("✓ Task 3 working\n")
    except Exception as e:
        print(f"✗ Task 3 error: {e}\n")
    
    # Test Task 4
    print("Task 4 - Literal Types:")
    try:
        set_config("database_url", "postgresql://localhost/db")
        set_config("debug_mode", True)
        set_config("api_timeout", 30)
        
        url = get_config("database_url")
        debug = get_config("debug_mode")
        timeout = get_config("api_timeout")
        
        print(f"Database URL: {url} (type: {type(url).__name__})")
        print(f"Debug mode: {debug} (type: {type(debug).__name__})")
        print(f"API timeout: {timeout} (type: {type(timeout).__name__})")
        print("✓ Task 4 working\n")
    except Exception as e:
        print(f"✗ Task 4 error: {e}\n")
    
    # Test Task 5
    print("Task 5 - Data Classes:")
    try:
        user = User(1, "Alice", "alice@example.com", 25)
        user.add_tag("admin")
        user.add_tag("user")
        removed = user.remove_tag("user")
        data = user.to_dict()
        print(f"User: {user.name}, Tags: {user.tags}")
        print(f"Removed tag: {removed}")
        print(f"User dict: {data}")
        print("✓ Task 5 working\n")
    except Exception as e:
        print(f"✗ Task 5 error: {e}\n")

if __name__ == "__main__":
    test_all_tasks()
