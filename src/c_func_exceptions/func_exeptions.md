# Functions and Exceptions

Functions are the building blocks of Python programs, allowing you to organize code into reusable, testable units. Exception handling is crucial for writing robust applications that can gracefully handle errors and unexpected situations. Together, they form the foundation of well-structured Python applications.

## Functions: The Heart of Python Programming

Functions in Python are first-class objects, meaning they can be assigned to variables, passed as arguments, and returned from other functions. This flexibility makes Python incredibly powerful for both simple scripts and complex applications.

## 1. Function Basics and Best Practices

A well-designed function should have a single responsibility, clear parameters, and predictable behavior. Good functions are easy to test, understand, and maintain.

### Basic Function Structure
```python
def calculate_area(length, width):
    """Calculate the area of a rectangle.
    
    Args:
        length (float): The length of the rectangle
        width (float): The width of the rectangle
        
    Returns:
        float: The area of the rectangle
    """
    return length * width
```

The docstring follows the Google style and clearly explains what the function does, its parameters, and return value.

### Function Arguments: Flexibility and Clarity

Python offers several ways to pass arguments to functions, each serving different use cases and improving code readability.

### Positional and Keyword Arguments
```python
def create_user(name, email, age=None, active=True):
    """Create a user with required and optional parameters."""
    user = {
        'name': name,
        'email': email,
        'age': age,
        'active': active
    }
    return user

# Different ways to call the function
user1 = create_user("Alice", "alice@example.com")
user2 = create_user("Bob", "bob@example.com", age=25)
user3 = create_user("Charlie", email="charlie@example.com", age=30, active=False)
```

Using keyword arguments makes function calls more readable and less prone to errors when you have many parameters.

## 2. Advanced Function Features

### *args and **kwargs for Flexible Functions
```python
def log_message(level, message, *details, **metadata):
    """Log a message with optional details and metadata.
    
    Args:
        level (str): Log level (INFO, WARNING, ERROR)
        message (str): Main log message
        *details: Additional details to include
        **metadata: Key-value pairs for extra context
    """
    print(f"[{level}] {message}")
    
    if details:
        print("Details:", ", ".join(str(d) for d in details))
    
    if metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        print(f"Metadata: {meta_str}")

# Usage examples
log_message("INFO", "User logged in")
log_message("WARNING", "Low disk space", "Only 5GB remaining", user_id=123, server="web-01")
```

This pattern is extremely useful for creating APIs and wrapper functions that need to handle varying numbers of arguments.

### Function Decorators: Enhancing Functions
```python
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def retry(max_attempts=3):
    """Decorator to retry a function on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            return None
        return wrapper
    return decorator

@timing_decorator
@retry(max_attempts=3)
def fetch_data(url):
    """Simulate fetching data that might fail."""
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network error")
    return f"Data from {url}"
```

Decorators allow you to add functionality to functions without modifying their code, following the principle of separation of concerns.

## 3. Exception Handling: Building Robust Applications

Exception handling is not just about preventing crashes—it's about creating predictable, user-friendly behavior when things go wrong. Good exception handling makes your applications more reliable and easier to debug.

### The Exception Hierarchy

Understanding Python's exception hierarchy helps you catch the right exceptions at the right level:

```python
# More specific exceptions should be caught first
def process_user_input(user_input):
    """Process user input with comprehensive error handling."""
    try:
        # Convert to integer
        number = int(user_input)
        
        # Perform calculation that might divide by zero
        result = 100 / number
        
        # Simulate file operation
        with open(f"data_{number}.txt", 'r') as file:
            content = file.read()
            
        return result, content
        
    except ValueError:
        # Handle conversion errors
        raise ValueError(f"'{user_input}' is not a valid number")
    except ZeroDivisionError:
        # Handle division by zero
        raise ValueError("Number cannot be zero")
    except FileNotFoundError:
        # Handle missing files
        raise FileNotFoundError(f"Data file for {user_input} not found")
    except PermissionError:
        # Handle permission issues
        raise PermissionError("Cannot access the data file")
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"Unexpected error processing {user_input}: {e}")
```

The order of except blocks matters—more specific exceptions should be caught before more general ones.

### Custom Exceptions: Making Errors Meaningful

Creating custom exceptions makes your code more expressive and helps other developers understand what went wrong:

```python
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

class AuthenticationError(Exception):
    """Raised when user authentication fails."""
    pass

class BusinessLogicError(Exception):
    """Raised when business rules are violated."""
    pass

class UserService:
    def __init__(self):
        self.users = {}
    
    def create_user(self, username, email, age):
        """Create a new user with validation."""
        # Validate input
        if not username or len(username) < 3:
            raise ValidationError("Username must be at least 3 characters long")
        
        if '@' not in email:
            raise ValidationError("Invalid email format")
        
        if age < 0 or age > 150:
            raise ValidationError("Age must be between 0 and 150")
        
        # Check business rules
        if username in self.users:
            raise BusinessLogicError(f"User '{username}' already exists")
        
        # Create user
        self.users[username] = {
            'email': email,
            'age': age,
            'created_at': time.time()
        }
        
        return f"User '{username}' created successfully"
    
    def authenticate_user(self, username, password):
        """Authenticate a user."""
        if username not in self.users:
            raise AuthenticationError("User not found")
        
        # Simulate password check
        if len(password) < 8:
            raise AuthenticationError("Invalid password")
        
        return f"User '{username}' authenticated successfully"
```

Custom exceptions make error handling more precise and help distinguish between different types of problems.

## 4. Exception Handling Best Practices

### The EAFP Principle: Easier to Ask for Forgiveness than Permission

Python encourages the EAFP (Easier to Ask for Forgiveness than Permission) approach over LBYL (Look Before You Leap):

### LBYL Approach (Not Recommended)
```python
def get_user_age(users, username):
    """Get user age using LBYL approach."""
    if username in users:
        if 'age' in users[username]:
            if isinstance(users[username]['age'], int):
                return users[username]['age']
    return None
```

### EAFP Approach (Recommended)
```python
def get_user_age(users, username):
    """Get user age using EAFP approach."""
    try:
        return users[username]['age']
    except (KeyError, TypeError):
        return None
```

The EAFP approach is more Pythonic because it's cleaner, faster in the common case, and handles race conditions better.

### Context Managers for Resource Management

Context managers ensure proper cleanup even when exceptions occur:

```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def database_connection(db_path):
    """Context manager for database connections."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        yield conn
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        raise DatabaseError(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed")

class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass

def save_user_data(db_path, user_data):
    """Save user data with proper error handling."""
    try:
        with database_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)",
                (user_data['name'], user_data['email'])
            )
            conn.commit()
            return "User data saved successfully"
    
    except DatabaseError:
        # Re-raise database-specific errors
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise RuntimeError(f"Failed to save user data: {e}")
```

Context managers guarantee cleanup operations, making your code more robust and preventing resource leaks.

## 5. Exception Chaining and Information Preservation

When handling exceptions, you often want to provide additional context while preserving the original error information:

```python
def process_data_file(filename):
    """Process a data file with detailed error information."""
    try:
        with open(filename, 'r') as file:
            content = file.read()
        
        # Process the content
        data = parse_json_data(content)
        return validate_and_transform(data)
    
    except FileNotFoundError as e:
        # Chain exceptions to preserve original error
        raise ProcessingError(f"Cannot process '{filename}': file not found") from e
    
    except json.JSONDecodeError as e:
        # Provide context about what was being parsed
        raise ProcessingError(f"Invalid JSON in '{filename}' at line {e.lineno}") from e
    
    except ValidationError as e:
        # Add file context to validation errors
        raise ProcessingError(f"Data validation failed in '{filename}': {e}") from e

class ProcessingError(Exception):
    """Raised when file processing fails."""
    pass

def parse_json_data(content):
    """Parse JSON content."""
    import json
    return json.loads(content)

def validate_and_transform(data):
    """Validate and transform data."""
    if not isinstance(data, dict):
        raise ValidationError("Data must be a dictionary")
    
    required_fields = ['name', 'email']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")
    
    return data
```

Exception chaining with `raise ... from ...` preserves the original exception while adding context, making debugging much easier.

## 6. Testing Functions and Exception Handling

Well-designed functions are easy to test, and proper exception handling should be thoroughly tested:

```python
import pytest

def test_create_user_success():
    """Test successful user creation."""
    service = UserService()
    result = service.create_user("alice", "alice@example.com", 25)
    assert "User 'alice' created successfully" in result
    assert "alice" in service.users

def test_create_user_validation_errors():
    """Test validation error handling."""
    service = UserService()
    
    # Test short username
    with pytest.raises(ValidationError, match="Username must be at least 3 characters"):
        service.create_user("ab", "alice@example.com", 25)
    
    # Test invalid email
    with pytest.raises(ValidationError, match="Invalid email format"):
        service.create_user("alice", "invalid-email", 25)
    
    # Test invalid age
    with pytest.raises(ValidationError, match="Age must be between 0 and 150"):
        service.create_user("alice", "alice@example.com", -5)

def test_duplicate_user():
    """Test business logic error for duplicate users."""
    service = UserService()
    service.create_user("alice", "alice@example.com", 25)
    
    with pytest.raises(BusinessLogicError, match="User 'alice' already exists"):
        service.create_user("alice", "different@example.com", 30)
```

Testing exception handling ensures your error cases work correctly and provide meaningful error messages.

## Key Takeaways

**Design Functions with Single Responsibility**: Each function should do one thing well. This makes testing easier and code more maintainable.

**Use Type Hints and Docstrings**: They serve as documentation and help catch errors early during development.

**Handle Exceptions at the Right Level**: Catch exceptions where you can meaningfully handle them, not just to suppress them.

**Create Custom Exceptions**: They make your code more expressive and help distinguish between different error conditions.

**Follow the EAFP Principle**: It's more Pythonic to try an operation and handle exceptions than to check conditions first.

**Use Context Managers**: They ensure proper resource cleanup even when exceptions occur.

**Preserve Exception Information**: Use exception chaining to maintain debugging information while adding context.

**Test Both Success and Failure Cases**: Comprehensive testing includes verifying that your functions handle errors correctly.

Remember: Good exception handling is not about preventing all errors—it's about handling errors gracefully and providing meaningful feedback when things go wrong. Functions should be predictable, well-documented, and robust in the face of unexpected inputs or conditions.