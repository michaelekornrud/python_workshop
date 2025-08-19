"""
Modules, Typing, and Tools Workshop - SOLUTIONS

Complete solutions for all tasks demonstrating module organization, type hints, and development tools.
"""

import math
import random
from typing import Any, Protocol, TypeVar, Generic, Literal, overload, TypeGuard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# =============================================================================
# TASK 1: Basic Module Organization - SOLUTION
# =============================================================================

# Module constants
PI: float = 3.14159265359
GOLDEN_RATIO: float = 1.618033988749

def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle.
    
    Args:
        radius: The radius of the circle
        
    Returns:
        The area of the circle
        
    Example:
        >>> calculate_circle_area(5.0)
        78.53981633974483
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return PI * radius * radius

def calculate_rectangle_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle.
    
    Args:
        length: The length of the rectangle
        width: The width of the rectangle
        
    Returns:
        The area of the rectangle
        
    Example:
        >>> calculate_rectangle_area(4.0, 6.0)
        24.0
    """
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return length * width

def calculate_distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points.
    
    Args:
        point1: First point as (x, y) coordinates
        point2: Second point as (x, y) coordinates
        
    Returns:
        The Euclidean distance between the points
        
    Example:
        >>> calculate_distance((0, 0), (3, 4))
        5.0
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# =============================================================================
# TASK 2: Advanced Type Hints with Generics - SOLUTION
# =============================================================================

T = TypeVar('T')

class Repository(Generic[T]):
    """Generic repository for storing and retrieving items of any type.
    
    Example:
        >>> repo = Repository[str]()
        >>> repo.save("1", "Hello")
        >>> repo.get("1")
        'Hello'
    """
    
    def __init__(self) -> None:
        """Initialize an empty repository."""
        self._data: dict[str, T] = {}
    
    def save(self, id: str, item: T) -> None:
        """Save an item with the given ID.
        
        Args:
            id: Unique identifier for the item
            item: The item to save
        """
        self._data[id] = item
    
    def get(self, id: str) -> T | None:
        """Retrieve an item by ID.
        
        Args:
            id: The ID of the item to retrieve
            
        Returns:
            The item if found, None otherwise
        """
        return self._data.get(id)
    
    def get_all(self) -> list[T]:
        """Get all items in the repository.
        
        Returns:
            List of all stored items
        """
        return list(self._data.values())
    
    def delete(self, id: str) -> bool:
        """Delete an item by ID.
        
        Args:
            id: The ID of the item to delete
            
        Returns:
            True if item was deleted, False if not found
        """
        if id in self._data:
            del self._data[id]
            return True
        return False

# =============================================================================
# TASK 3: Protocol-Based Design - SOLUTION
# =============================================================================

class Drawable(Protocol):
    """Protocol for objects that can be drawn and have an area."""
    
    def draw(self) -> str:
        """Draw the object and return a string representation."""
        ...
    
    def get_area(self) -> float:
        """Get the area of the object."""
        ...

class Circle:
    """A circle that implements the Drawable protocol."""
    
    def __init__(self, radius: float) -> None:
        """Initialize a circle with given radius."""
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self.radius = radius
    
    def draw(self) -> str:
        """Draw the circle."""
        return f"Circle with radius {self.radius}"
    
    def get_area(self) -> float:
        """Get the area of the circle."""
        return PI * self.radius * self.radius

class Rectangle:
    """A rectangle that implements the Drawable protocol."""
    
    def __init__(self, length: float, width: float) -> None:
        """Initialize a rectangle with given dimensions."""
        if length <= 0 or width <= 0:
            raise ValueError("Length and width must be positive")
        self.length = length
        self.width = width
    
    def draw(self) -> str:
        """Draw the rectangle."""
        return f"Rectangle {self.length}x{self.width}"
    
    def get_area(self) -> float:
        """Get the area of the rectangle."""
        return self.length * self.width

class Triangle:
    """A triangle that implements the Drawable protocol."""
    
    def __init__(self, base: float, height: float) -> None:
        """Initialize a triangle with given base and height."""
        if base <= 0 or height <= 0:
            raise ValueError("Base and height must be positive")
        self.base = base
        self.height = height
    
    def draw(self) -> str:
        """Draw the triangle."""
        return f"Triangle with base {self.base} and height {self.height}"
    
    def get_area(self) -> float:
        """Get the area of the triangle."""
        return 0.5 * self.base * self.height

def render_shapes(shapes: list[Drawable]) -> list[str]:
    """Render a list of drawable shapes.
    
    Args:
        shapes: List of objects implementing the Drawable protocol
        
    Returns:
        List of string representations of the shapes
    """
    return [shape.draw() for shape in shapes]

# =============================================================================
# TASK 4: Literal Types and Overloads - SOLUTION
# =============================================================================

ConfigKey = Literal["database_url", "debug_mode", "api_timeout"]

# Global configuration storage
_config: dict[str, str | bool | int] = {
    "database_url": "sqlite:///default.db",
    "debug_mode": False,
    "api_timeout": 30
}

@overload
def get_config(key: Literal["database_url"]) -> str: ...

@overload  
def get_config(key: Literal["debug_mode"]) -> bool: ...

@overload
def get_config(key: Literal["api_timeout"]) -> int: ...

def get_config(key: ConfigKey) -> str | bool | int:
    """Get configuration value by key.
    
    Args:
        key: The configuration key
        
    Returns:
        The configuration value with the correct type
        
    Raises:
        KeyError: If the key is not found
    """
    if key not in _config:
        raise KeyError(f"Configuration key '{key}' not found")
    return _config[key]

def set_config(key: ConfigKey, value: str | bool | int) -> None:
    """Set configuration value by key.
    
    Args:
        key: The configuration key
        value: The value to set
        
    Raises:
        TypeError: If the value type doesn't match the expected type for the key
    """
    expected_types = {
        "database_url": str,
        "debug_mode": bool,
        "api_timeout": int
    }
    
    expected_type = expected_types[key]
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected {expected_type.__name__} for key '{key}', got {type(value).__name__}")
    
    _config[key] = value

# =============================================================================
# TASK 5: Data Classes with Validation - SOLUTION
# =============================================================================

@dataclass
class User:
    """User data class with validation.
    
    Attributes:
        id: Unique user identifier
        name: User's full name
        email: User's email address
        age: User's age
        tags: List of user tags
    """
    id: int
    name: str
    email: str
    age: int
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate user data after initialization."""
        if "@" not in self.email:
            raise ValueError("Email must contain '@' symbol")
        
        if not (0 <= self.age <= 150):
            raise ValueError("Age must be between 0 and 150")
        
        if self.id <= 0:
            raise ValueError("ID must be positive")
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the user.
        
        Args:
            tag: The tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the user.
        
        Args:
            tag: The tag to remove
            
        Returns:
            True if tag was removed, False if not found
        """
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert user to dictionary.
        
        Returns:
            Dictionary representation of the user
        """
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "tags": self.tags.copy()
        }

# =============================================================================
# TASK 6: Module Package Structure - SOLUTION
# =============================================================================

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, id: int) -> None:
        """Initialize the model with an ID."""
        self.id = id
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the model data."""
        pass
    
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        pass

class Product(BaseModel):
    """Product model representing items for sale."""
    
    def __init__(self, id: int, name: str, price: float) -> None:
        """Initialize a product."""
        super().__init__(id)
        self.name = name
        self.price = price
    
    def validate(self) -> bool:
        """Validate product data."""
        return len(self.name) > 0 and self.price >= 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert product to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "type": "product"
        }

class Order(BaseModel):
    """Order model representing customer orders."""
    
    def __init__(self, id: int, customer_id: int, total: float) -> None:
        """Initialize an order."""
        super().__init__(id)
        self.customer_id = customer_id
        self.total = total
    
    def validate(self) -> bool:
        """Validate order data."""
        return self.customer_id > 0 and self.total >= 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "total": self.total,
            "type": "order"
        }

def get_all_models() -> list[type[BaseModel]]:
    """Get all model classes.
    
    Returns:
        List of all model class types
    """
    return [Product, Order]

# Public API
__all__ = ["BaseModel", "Product", "Order", "get_all_models"]

# =============================================================================
# TASK 7: Error Handling with Custom Types - SOLUTION
# =============================================================================

@dataclass
class Success(Generic[T]):
    """Represents a successful result with a value."""
    value: T

@dataclass
class Error:
    """Represents an error result with a message."""
    message: str

Result = Success[T] | Error

def divide_safely(a: float, b: float) -> Result[float]:
    """Safely divide two numbers.
    
    Args:
        a: The dividend
        b: The divisor
        
    Returns:
        Success with the result or Error with message
    """
    if b == 0:
        return Error("Division by zero")
    return Success(a / b)

def process_results(results: list[Result[float]]) -> list[float]:
    """Extract successful values from a list of results.
    
    Args:
        results: List of Result objects
        
    Returns:
        List of successful float values
    """
    successful_values = []
    for result in results:
        if isinstance(result, Success):
            successful_values.append(result.value)
    return successful_values

# =============================================================================
# TASK 8: Type Guards and Runtime Type Checking - SOLUTION
# =============================================================================

def is_string_list(obj: Any) -> TypeGuard[list[str]]:
    """Check if an object is a list of strings.
    
    Args:
        obj: Object to check
        
    Returns:
        True if obj is a list of strings
    """
    return (
        isinstance(obj, list) and
        all(isinstance(item, str) for item in obj)
    )

def is_user_dict(obj: Any) -> TypeGuard[dict[str, Any]]:
    """Check if an object is a valid user dictionary.
    
    Args:
        obj: Object to check
        
    Returns:
        True if obj is a valid user dict
    """
    if not isinstance(obj, dict):
        return False
    
    required_fields = {"name", "age"}
    return (
        all(field in obj for field in required_fields) and
        isinstance(obj.get("name"), str) and
        isinstance(obj.get("age"), int)
    )

def process_mixed_data(data: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Process mixed data and separate strings and user dicts.
    
    Args:
        data: List of mixed data
        
    Returns:
        Tuple of (string list, user dict list)
    """
    strings = []
    user_dicts = []
    
    for item in data:
        if isinstance(item, str):
            strings.append(item)
        elif is_user_dict(item):
            user_dicts.append(item)
    
    return strings, user_dicts

# =============================================================================
# TASK 9: Comprehensive Package with Testing Support - SOLUTION
# =============================================================================

class Dataset:
    """A dataset for statistical calculations."""
    
    def __init__(self) -> None:
        """Initialize an empty dataset."""
        self._values: list[float] = []
    
    def add_value(self, value: float) -> None:
        """Add a value to the dataset.
        
        Args:
            value: The value to add
        """
        self._values.append(value)
    
    def mean(self) -> float | None:
        """Calculate the mean of the dataset.
        
        Returns:
            The mean value or None if dataset is empty
        """
        if not self._values:
            return None
        return sum(self._values) / len(self._values)
    
    def median(self) -> float | None:
        """Calculate the median of the dataset.
        
        Returns:
            The median value or None if dataset is empty
        """
        if not self._values:
            return None
        
        sorted_values = sorted(self._values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            # Even number of values
            mid1 = sorted_values[n // 2 - 1]
            mid2 = sorted_values[n // 2]
            return (mid1 + mid2) / 2
        else:
            # Odd number of values
            return sorted_values[n // 2]
    
    def std_dev(self) -> float | None:
        """Calculate the standard deviation of the dataset.
        
        Returns:
            The standard deviation or None if dataset is empty
        """
        if not self._values:
            return None
        
        mean_val = self.mean()
        if mean_val is None:
            return None
        
        variance = sum((x - mean_val) ** 2 for x in self._values) / len(self._values)
        return math.sqrt(variance)

def create_sample_dataset(size: int, seed: int | None = None) -> Dataset:
    """Create a dataset with random sample values.
    
    Args:
        size: Number of values to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dataset with random values
        
    Raises:
        ValueError: If size is not positive
    """
    if size <= 0:
        raise ValueError("Size must be positive")
    
    if seed is not None:
        random.seed(seed)
    
    dataset = Dataset()
    for _ in range(size):
        # Generate random values between 0 and 100
        value = random.uniform(0, 100)
        dataset.add_value(value)
    
    return dataset

# =============================================================================
# BONUS TASK: Advanced Type System Features - SOLUTION
# =============================================================================

class ConfigValue(Protocol[T]):
    """Protocol for configuration values."""
    
    def get(self) -> T:
        """Get the configuration value."""
        ...

class StringConfig:
    """Configuration for string values."""
    
    def __init__(self, value: str) -> None:
        """Initialize with a string value."""
        self._value = value
    
    def get(self) -> str:
        """Get the string value."""
        return self._value

class IntConfig:
    """Configuration for integer values."""
    
    def __init__(self, value: int) -> None:
        """Initialize with an integer value."""
        self._value = value
    
    def get(self) -> int:
        """Get the integer value."""
        return self._value

class BoolConfig:
    """Configuration for boolean values."""
    
    def __init__(self, value: bool) -> None:
        """Initialize with a boolean value."""
        self._value = value
    
    def get(self) -> bool:
        """Get the boolean value."""
        return self._value

class ConfigManager:
    """Type-safe configuration manager."""
    
    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self._configs: dict[str, Any] = {}
    
    def register_config(self, key: str, config: ConfigValue[Any]) -> None:
        """Register a configuration value.
        
        Args:
            key: Configuration key
            config: Configuration value object
        """
        self._configs[key] = config
    
    def get_config(self, key: str) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            
        Returns:
            The configuration value
            
        Raises:
            KeyError: If key is not found
        """
        if key not in self._configs:
            raise KeyError(f"Configuration key '{key}' not found")
        
        config = self._configs[key]
        return config.get()

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_all_solutions():
    """Test all implemented solutions"""
    print("Testing all Modules, Typing, and Tools solutions...\n")
    
    # Test Task 1
    print("Task 1 - Math Utilities:")
    area = calculate_circle_area(5.0)
    rect_area = calculate_rectangle_area(4.0, 6.0)
    distance = calculate_distance((0, 0), (3, 4))
    print(f"Circle area (r=5): {area:.2f}")
    print(f"Rectangle area (4x6): {rect_area}")
    print(f"Distance (0,0) to (3,4): {distance}")
    print("✓ Task 1 complete\n")
    
    # Test Task 2
    print("Task 2 - Generic Repository:")
    repo = Repository[str]()
    repo.save("1", "Hello")
    repo.save("2", "World")
    item = repo.get("1")
    all_items = repo.get_all()
    deleted = repo.delete("1")
    print(f"Retrieved item: {item}")
    print(f"All items: {all_items}")
    print(f"Deleted successfully: {deleted}")
    print("✓ Task 2 complete\n")
    
    # Test Task 3
    print("Task 3 - Protocol Design:")
    circle = Circle(5.0)
    rectangle = Rectangle(4.0, 6.0)
    triangle = Triangle(3.0, 4.0)
    shapes = [circle, rectangle, triangle]
    rendered = render_shapes(shapes)
    for shape_desc in rendered:
        print(f"  {shape_desc}")
    print("✓ Task 3 complete\n")
    
    # Test Task 4
    print("Task 4 - Literal Types:")
    set_config("database_url", "postgresql://localhost/db")
    set_config("debug_mode", True)
    set_config("api_timeout", 30)
    
    url = get_config("database_url")
    debug = get_config("debug_mode")
    timeout = get_config("api_timeout")
    
    print(f"Database URL: {url} (type: {type(url).__name__})")
    print(f"Debug mode: {debug} (type: {type(debug).__name__})")
    print(f"API timeout: {timeout} (type: {type(timeout).__name__})")
    print("✓ Task 4 complete\n")
    
    # Test Task 5
    print("Task 5 - Data Classes:")
    user = User(1, "Alice", "alice@example.com", 25)
    user.add_tag("admin")
    user.add_tag("user")
    removed = user.remove_tag("user")
    data = user.to_dict()
    print(f"User: {user.name}, Age: {user.age}, Tags: {user.tags}")
    print(f"Removed tag: {removed}")
    print("✓ Task 5 complete\n")
    
    # Test Task 6
    print("Task 6 - Package Structure:")
    models = get_all_models()
    product = Product(1, "Laptop", 999.99)
    order = Order(1, 123, 1999.99)
    print(f"Available models: {[m.__name__ for m in models]}")
    print(f"Product valid: {product.validate()}")
    print(f"Order valid: {order.validate()}")
    print("✓ Task 6 complete\n")
    
    # Test Task 7
    print("Task 7 - Result Types:")
    result1 = divide_safely(10, 2)
    result2 = divide_safely(10, 0)
    results = [Success(1.0), Error("failed"), Success(2.0), Success(3.0)]
    values = process_results(results)
    print(f"10 / 2 = {result1}")
    print(f"10 / 0 = {result2}")
    print(f"Successful values: {values}")
    print("✓ Task 7 complete\n")
    
    # Test Task 8
    print("Task 8 - Type Guards:")
    mixed_data = ["hello", {"name": "Alice", "age": 25}, "world", {"invalid": "data"}, {"name": "Bob", "age": 30}]
    strings, users = process_mixed_data(mixed_data)
    print(f"Strings: {strings}")
    print(f"Valid user dicts: {users}")
    print("✓ Task 8 complete\n")
    
    # Test Task 9
    print("Task 9 - Statistics Package:")
    dataset = Dataset()
    for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
        dataset.add_value(value)
    
    print(f"Mean: {dataset.mean()}")
    print(f"Median: {dataset.median()}")
    print(f"Std Dev: {dataset.std_dev():.3f}")
    
    sample = create_sample_dataset(5, seed=42)
    print(f"Sample dataset mean: {sample.mean():.2f}")
    print("✓ Task 9 complete\n")

if __name__ == "__main__":
    test_all_solutions()
