"""
Design Patterns Workshop - Advanced Python Patterns for Enterprise Developers

This workshop demonstrates how classic Gang of Four design patterns are implemented
in Python, with specific focus on how they differ from Java/C# implementations.
Python's dynamic nature often provides more elegant solutions.

Complete the following tasks to master advanced Python design patterns.
"""

import abc
import threading
import weakref
from collections.abc import Callable
from typing import Any, TypeVar

# =============================================================================
# TASK 1: Singleton Pattern - Python vs Java/C#
# =============================================================================

"""
TASK 1: Implement Singleton Pattern (Multiple Approaches)

In Java/C#, Singleton is typically implemented with static fields and methods.
Python offers several approaches, each with different trade-offs.

Requirements:
- Implement 4 different Singleton approaches:
  1. Classic Singleton with __new__
  2. Decorator-based Singleton
  3. Metaclass-based Singleton  
  4. Module-level Singleton (most Pythonic)
- Thread-safe implementations
- Compare performance and readability

Example usage:
config = ConfigManager()
config2 = ConfigManager()
assert config is config2  # Same instance
"""

# Approach 1: Classic Singleton with __new__
class ClassicSingleton:
    """Classic singleton using __new__ method"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        # Your implementation here
        pass
    
    def __init__(self, name: str = "default"):
        # Your implementation here
        pass

# Approach 2: Decorator-based Singleton
def singleton_decorator(cls):
    """Decorator to make any class a singleton"""
    # Your implementation here
    pass

@singleton_decorator
class DecoratorSingleton:
    """Singleton using decorator pattern"""
    def __init__(self, config_path: str = "config.ini"):
        # Your implementation here
        pass

# Approach 3: Metaclass-based Singleton
class SingletonMeta(type):
    """Metaclass that creates singleton instances"""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        # Your implementation here
        pass

class MetaclassSingleton(metaclass=SingletonMeta):
    """Singleton using metaclass"""
    def __init__(self, database_url: str = "sqlite:///default.db"):
        # Your implementation here
        pass

# Approach 4: Module-level Singleton (Most Pythonic)
class _ConfigManager:
    """Private config manager class"""
    def __init__(self):
        # Your implementation here
        pass
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        # Your implementation here
        pass
    
    def set_setting(self, key: str, value: Any) -> None:
        # Your implementation here
        pass

# Module-level instance (most Pythonic approach)
config_manager = _ConfigManager()

# =============================================================================
# TASK 2: Factory Pattern - Dynamic Python Style
# =============================================================================

"""
TASK 2: Implement Factory Pattern with Python Features

Java/C# factories often use switch statements or if-else chains.
Python's dynamic features allow for more elegant factory implementations.

Requirements:
- Create a shape factory using multiple approaches
- Registry-based factory (most Pythonic)
- Class-based factory with dynamic class loading
- Abstract factory for related object families
- Factory with configuration and validation

Example usage:
factory = ShapeFactory()
circle = factory.create_shape("circle", radius=5)
square = factory.create_shape("square", side=4)
"""

# Base shape interface
class Shape(abc.ABC):
    """Abstract base class for all shapes"""
    
    @abc.abstractmethod
    def area(self) -> float:
        """Calculate the area of the shape"""
        pass
    
    @abc.abstractmethod
    def perimeter(self) -> float:
        """Calculate the perimeter of the shape"""
        pass
    
    @abc.abstractmethod
    def draw(self) -> str:
        """Return a string representation of the shape"""
        pass

# Concrete shape implementations
class Circle(Shape):
    """Circle implementation"""
    def __init__(self, radius: float):
        # Your implementation here
        pass
    
    def area(self) -> float:
        # Your implementation here
        pass
    
    def perimeter(self) -> float:
        # Your implementation here
        pass
    
    def draw(self) -> str:
        # Your implementation here
        pass

class Rectangle(Shape):
    """Rectangle implementation"""
    def __init__(self, width: float, height: float):
        # Your implementation here
        pass
    
    def area(self) -> float:
        # Your implementation here
        pass
    
    def perimeter(self) -> float:
        # Your implementation here
        pass
    
    def draw(self) -> str:
        # Your implementation here
        pass

class Triangle(Shape):
    """Triangle implementation"""
    def __init__(self, base: float, height: float, side1: float, side2: float):
        # Your implementation here
        pass
    
    def area(self) -> float:
        # Your implementation here
        pass
    
    def perimeter(self) -> float:
        # Your implementation here
        pass
    
    def draw(self) -> str:
        # Your implementation here
        pass

# Registry-based Factory (Most Pythonic)
class ShapeFactory:
    """Registry-based shape factory"""
    _shapes: dict[str, type] = {}
    
    @classmethod
    def register_shape(cls, name: str, shape_class: type) -> None:
        """Register a new shape type"""
        # Your implementation here
        pass
    
    @classmethod
    def create_shape(cls, shape_type: str, **kwargs) -> Shape:
        """Create a shape instance"""
        # Your implementation here
        pass
    
    @classmethod
    def list_available_shapes(cls) -> list[str]:
        """List all registered shape types"""
        # Your implementation here
        pass

# Auto-register shapes using decorator
def register_shape(name: str):
    """Decorator to auto-register shapes"""
    def decorator(cls):
        # Your implementation here
        return cls
    return decorator

# Abstract Factory for UI themes
class UITheme(abc.ABC):
    """Abstract factory for UI themes"""
    
    @abc.abstractmethod
    def create_button(self, text: str) -> "Button":
        pass
    
    @abc.abstractmethod
    def create_window(self, title: str) -> "Window":
        pass

class Button(abc.ABC):
    @abc.abstractmethod
    def render(self) -> str:
        pass

class Window(abc.ABC):
    @abc.abstractmethod
    def show(self) -> str:
        pass

# Concrete theme implementations
class DarkTheme(UITheme):
    """Dark theme implementation"""
    def create_button(self, text: str) -> Button:
        # Your implementation here
        pass
    
    def create_window(self, title: str) -> Window:
        # Your implementation here
        pass

class LightTheme(UITheme):
    """Light theme implementation"""
    def create_button(self, text: str) -> Button:
        # Your implementation here
        pass
    
    def create_window(self, title: str) -> Window:
        # Your implementation here
        pass

# =============================================================================
# TASK 3: Observer Pattern - Python Event System
# =============================================================================

"""
TASK 3: Implement Observer Pattern with Python Features

Java/C# observer patterns often use interfaces and event handlers.
Python can use decorators, descriptors, and weak references for cleaner implementations.

Requirements:
- Event system with decorators
- Weak reference observers (avoid memory leaks)
- Type-safe event handlers
- Async observer support
- Property observers using descriptors

Example usage:
@event_handler("user_created")
def send_welcome_email(user_data):
    print(f"Sending welcome email to {user_data['email']}")

user_service = UserService()
user_service.create_user("john@example.com")  # Triggers event
"""

T = TypeVar('T')

class Event:
    """Event class for type-safe events"""
    def __init__(self, name: str):
        # Your implementation here
        pass

class EventBus:
    """Central event bus for managing observers"""
    _instance = None
    _handlers: dict[str, list[weakref.WeakMethod]] = {}
    
    def __new__(cls):
        # Your implementation here
        pass
    
    def subscribe(self, event_name: str, handler: Callable) -> None:
        """Subscribe to an event"""
        # Your implementation here
        pass
    
    def unsubscribe(self, event_name: str, handler: Callable) -> None:
        """Unsubscribe from an event"""
        # Your implementation here
        pass
    
    def publish(self, event_name: str, **kwargs) -> None:
        """Publish an event to all subscribers"""
        # Your implementation here
        pass

def event_handler(event_name: str):
    """Decorator to register event handlers"""
    def decorator(func):
        # Your implementation here
        return func
    return decorator

# Property Observer using Descriptor
class ObservableProperty:
    """Descriptor that notifies observers when property changes"""
    def __init__(self, initial_value: Any = None, event_name: str = None):
        # Your implementation here
        pass
    
    def __get__(self, obj, objtype=None):
        # Your implementation here
        pass
    
    def __set__(self, obj, value):
        # Your implementation here
        pass

class UserService:
    """Service with observable properties"""
    user_count = ObservableProperty(0, "user_count_changed")
    
    def __init__(self):
        # Your implementation here
        pass
    
    def create_user(self, email: str) -> dict[str, Any]:
        """Create a new user and trigger events"""
        # Your implementation here
        pass

# =============================================================================
# TASK 4: Strategy Pattern - Functional vs OOP
# =============================================================================

"""
TASK 4: Implement Strategy Pattern (Functional and OOP)

Java/C# strategy pattern typically uses interfaces and classes.
Python allows both traditional OOP and functional approaches.

Requirements:
- Traditional OOP strategy pattern
- Functional strategy using first-class functions
- Strategy registry with decorators
- Context-aware strategies
- Async strategy support

Example usage:
calculator = Calculator()
calculator.set_strategy("add")
result = calculator.calculate(5, 3)  # 8

# Functional approach
result = calculate(5, 3, strategy="multiply")  # 15
"""

# Traditional OOP Strategy Pattern
class CalculationStrategy(abc.ABC):
    """Abstract strategy for calculations"""
    
    @abc.abstractmethod
    def calculate(self, a: float, b: float) -> float:
        """Perform calculation"""
        pass

class AddStrategy(CalculationStrategy):
    """Addition strategy"""
    def calculate(self, a: float, b: float) -> float:
        # Your implementation here
        pass

class SubtractStrategy(CalculationStrategy):
    """Subtraction strategy"""
    def calculate(self, a: float, b: float) -> float:
        # Your implementation here
        pass

class MultiplyStrategy(CalculationStrategy):
    """Multiplication strategy"""
    def calculate(self, a: float, b: float) -> float:
        # Your implementation here
        pass

class Calculator:
    """Calculator using strategy pattern"""
    def __init__(self):
        # Your implementation here
        pass
    
    def set_strategy(self, strategy: CalculationStrategy) -> None:
        """Set the calculation strategy"""
        # Your implementation here
        pass
    
    def calculate(self, a: float, b: float) -> float:
        """Perform calculation using current strategy"""
        # Your implementation here
        pass

# Functional Strategy Pattern
_strategies: dict[str, Callable[[float, float], float]] = {}

def strategy(name: str):
    """Decorator to register calculation strategies"""
    def decorator(func):
        # Your implementation here
        return func
    return decorator

@strategy("add")
def add_strategy(a: float, b: float) -> float:
    # Your implementation here
    pass

@strategy("subtract")
def subtract_strategy(a: float, b: float) -> float:
    # Your implementation here
    pass

@strategy("multiply")
def multiply_strategy(a: float, b: float) -> float:
    # Your implementation here
    pass

def calculate(a: float, b: float, strategy_name: str) -> float:
    """Functional approach to strategy pattern"""
    # Your implementation here
    pass

# =============================================================================
# TASK 5: Command Pattern - Functional Approach
# =============================================================================

"""
TASK 5: Implement Command Pattern with Functions

Java/C# command pattern uses command objects with execute methods.
Python's first-class functions provide a more natural implementation.

Requirements:
- Traditional command objects
- Functional command using closures
- Command queue with undo/redo
- Macro commands (composite)
- Async command execution

Example usage:
editor = TextEditor()
command = InsertTextCommand(editor, "Hello World")
invoker = CommandInvoker()
invoker.execute(command)
invoker.undo()  # Undoes the insert
"""

class Command(abc.ABC):
    """Abstract command interface"""
    
    @abc.abstractmethod
    def execute(self) -> None:
        """Execute the command"""
        pass
    
    @abc.abstractmethod
    def undo(self) -> None:
        """Undo the command"""
        pass

class TextEditor:
    """Simple text editor for command pattern demo"""
    def __init__(self):
        # Your implementation here
        pass
    
    def insert_text(self, text: str, position: int = None) -> None:
        """Insert text at position"""
        # Your implementation here
        pass
    
    def delete_text(self, start: int, length: int) -> str:
        """Delete text and return deleted content"""
        # Your implementation here
        pass
    
    def get_content(self) -> str:
        """Get current content"""
        # Your implementation here
        pass

class InsertTextCommand(Command):
    """Command to insert text"""
    def __init__(self, editor: TextEditor, text: str, position: int = None):
        # Your implementation here
        pass
    
    def execute(self) -> None:
        # Your implementation here
        pass
    
    def undo(self) -> None:
        # Your implementation here
        pass

class DeleteTextCommand(Command):
    """Command to delete text"""
    def __init__(self, editor: TextEditor, start: int, length: int):
        # Your implementation here
        pass
    
    def execute(self) -> None:
        # Your implementation here
        pass
    
    def undo(self) -> None:
        # Your implementation here
        pass

class CommandInvoker:
    """Command invoker with undo/redo support"""
    def __init__(self):
        # Your implementation here
        pass
    
    def execute(self, command: Command) -> None:
        """Execute a command"""
        # Your implementation here
        pass
    
    def undo(self) -> bool:
        """Undo the last command"""
        # Your implementation here
        pass
    
    def redo(self) -> bool:
        """Redo the last undone command"""
        # Your implementation here
        pass

# Functional Command Pattern
def create_command(execute_func: Callable, undo_func: Callable) -> Command:
    """Create a command from functions"""
    # Your implementation here
    pass

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_design_patterns():
    """Test all design pattern implementations"""
    print("Testing Design Patterns...")
    
    # Test Singleton patterns
    print("\n1. Testing Singleton Patterns:")
    # Your test implementation here
    
    # Test Factory pattern
    print("\n2. Testing Factory Pattern:")
    # Your test implementation here
    
    # Test Observer pattern
    print("\n3. Testing Observer Pattern:")
    # Your test implementation here
    
    # Test Strategy pattern
    print("\n4. Testing Strategy Pattern:")
    # Your test implementation here
    
    # Test Command pattern
    print("\n5. Testing Command Pattern:")
    # Your test implementation here
    
    print("\n✅ All design pattern tests completed!")

if __name__ == "__main__":
    test_design_patterns()
