"""
Functions and Exceptions Workshop - SOLUTIONS

Complete solutions for all tasks to demonstrate function design and exception handling.
"""

import json
import random
import time
from functools import wraps
from typing import Any

# =============================================================================
# TASK 1: Basic Function Design - SOLUTION
# =============================================================================

def calculate_positive_average(numbers: list[int | float]) -> float | None:
    """Calculate the average of positive numbers in a list.
    
    Args:
        numbers: A list of numeric values (int or float).
        
    Returns:
        The average of positive numbers, or None if no positive numbers exist.
        
    Examples:
        >>> calculate_positive_average([1, 2, 3, -1, -2])
        2.0
        >>> calculate_positive_average([-1, -2, -3])
        None
        >>> calculate_positive_average([])
        None
    """
    if not numbers:
        return None
    
    positive_numbers = [num for num in numbers if num > 0]
    
    if not positive_numbers:
        return None
        
    return sum(positive_numbers) / len(positive_numbers)

# =============================================================================
# TASK 2: User Creation with Flexible Parameters - SOLUTION
# =============================================================================

def create_user_profile(name: str, email: str, age: int | None = None, 
                       phone: str | None = None, active: bool = True, **kwargs) -> dict[str, Any]:
    """Create a user profile with required and optional information.
    
    Args:
        name: User's full name (required)
        email: User's email address (required)
        age: User's age (optional)
        phone: User's phone number (optional)
        active: Whether the user account is active (default: True)
        **kwargs: Additional optional fields
        
    Returns:
        Dictionary containing all user information including created_at timestamp
        
    Examples:
        >>> profile = create_user_profile("Alice", "alice@example.com")
        >>> profile['name']
        'Alice'
        >>> profile['active']
        True
    """
    user_profile = {
        'name': name,
        'email': email,
        'age': age,
        'phone': phone,
        'active': active,
        'created_at': time.time()
    }
    
    # Add any additional fields from kwargs
    user_profile.update(kwargs)
    
    return user_profile

# =============================================================================
# TASK 3: Flexible Calculator with *args and **kwargs - SOLUTION
# =============================================================================

def flexible_calculator(*args, **kwargs) -> int | float:
    """Perform operations on any number of numeric arguments.
    
    Args:
        *args: Any number of numeric arguments
        **kwargs: operation ("add", "multiply", "average"), round_to (int)
        
    Returns:
        Numeric result of the operation
        
    Raises:
        ValueError: If no arguments provided or invalid operation
        
    Examples:
        >>> flexible_calculator(1, 2, 3, 4, 5)
        15
        >>> flexible_calculator(2, 3, 4, operation="multiply")
        24
        >>> flexible_calculator(1.234, 5.678, operation="average", round_to=2)
        3.46
    """
    if not args:
        raise ValueError("At least one numeric argument is required")
    
    operation = kwargs.get("operation", "add")
    round_to = kwargs.get("round_to", None)
    
    if operation == "add":
        result = sum(args)
    elif operation == "multiply":
        result = 1
        for num in args:
            result *= num
    elif operation == "average":
        result = sum(args) / len(args)
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    if round_to is not None:
        return round(result, round_to)
    
    return result

# =============================================================================
# TASK 4: Input Validation Decorator - SOLUTION
# =============================================================================

def validate_input(input_type: str, min_val: int | float | None = None, 
                  max_val: int | float | None = None, required_char: str | None = None):
    """Decorator that validates function inputs based on specified criteria.
    
    Args:
        input_type: Type to validate ("int", "str", "float")
        min_val: Minimum value for numeric types
        max_val: Maximum value for numeric types
        required_char: Required character for string types
        
    Returns:
        Decorator function
        
    Raises:
        TypeError: If input type doesn't match expected type
        ValueError: If input value doesn't meet criteria
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                return func(*args, **kwargs)
            
            first_arg = args[0]
            
            # Type validation
            if input_type == "int" and not isinstance(first_arg, int):
                raise TypeError(f"Expected int, got {type(first_arg).__name__}")
            elif input_type == "str" and not isinstance(first_arg, str):
                raise TypeError(f"Expected str, got {type(first_arg).__name__}")
            elif input_type == "float" and not isinstance(first_arg, (int | float)):
                raise TypeError(f"Expected float, got {type(first_arg).__name__}")
            
            # Range validation for numbers
            if input_type in ("int", "float"):
                if min_val is not None and first_arg < min_val:
                    raise ValueError(f"Value {first_arg} is below minimum {min_val}")
                if max_val is not None and first_arg > max_val:
                    raise ValueError(f"Value {first_arg} is above maximum {max_val}")
            
            # Character requirement for strings
            if input_type == "str" and required_char is not None:
                if required_char not in first_arg:
                    raise ValueError(f"String must contain '{required_char}'")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example functions using the decorator
@validate_input("int", min_val=0, max_val=150)
def process_age(age: int) -> str:
    """Process age with decorator validation"""
    return f"Processing age: {age}"

@validate_input("str", required_char="@")
def process_email(email: str) -> str:
    """Process email with decorator validation"""
    return f"Processing email: {email}"

# =============================================================================
# TASK 5: Custom Exception Classes for Banking - SOLUTION
# =============================================================================

class BankError(Exception):
    """Base exception for banking operations"""
    pass

class InsufficientFundsError(BankError):
    """Raised when withdrawal amount exceeds available balance"""
    
    def __init__(self, amount: float, balance: float):
        self.amount = amount
        self.balance = balance
        super().__init__(f"Insufficient funds: attempted to withdraw {amount}, but only {balance} available")  # noqa : E501

class InvalidAmountError(BankError):
    """Raised when withdrawal amount is invalid (negative or zero)"""
    
    def __init__(self, amount: float):
        self.amount = amount
        super().__init__(f"Invalid withdrawal amount: {amount}. Amount must be positive")

class DailyLimitError(BankError):
    """Raised when withdrawal amount exceeds daily limit"""
    
    def __init__(self, amount: float, limit: float):
        self.amount = amount
        self.limit = limit
        super().__init__(f"Daily limit exceeded: attempted {amount}, limit is {limit}")

def withdraw_money(balance: float, amount: float, daily_limit: float = 10000) -> float:
    """Withdraw money from account with validation.
    
    Args:
        balance: Current account balance
        amount: Amount to withdraw
        daily_limit: Maximum daily withdrawal limit
        
    Returns:
        New balance after withdrawal
        
    Raises:
        InvalidAmountError: If amount is <= 0
        InsufficientFundsError: If amount > balance
        DailyLimitError: If amount > daily_limit
    """
    if amount <= 0:
        raise InvalidAmountError(amount)
    
    if amount > balance:
        raise InsufficientFundsError(amount, balance)
    
    if amount > daily_limit:
        raise DailyLimitError(amount, daily_limit)
    
    return balance - amount

# =============================================================================
# TASK 6: JSON File Processing with Error Handling - SOLUTION
# =============================================================================

def read_json_file(filename: str) -> dict[str, Any] | list[Any]:
    """Safely read and parse JSON files with comprehensive error handling.
    
    Args:
        filename: Path to the JSON file
        
    Returns:
        Parsed JSON data (dict or list)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be accessed
        ValueError: If JSON is invalid
    """
    try:
        with open(filename) as file:
            content = file.read()
            return json.loads(content)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"JSON file not found: {filename}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied accessing file: {filename}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filename}: {e}") from e

# =============================================================================
# TASK 7: User Data Processing with Exception Chaining - SOLUTION
# =============================================================================

class UserProcessingError(Exception):
    """Exception raised when user data processing fails"""
    pass

def process_user_data(user_data: dict[str, Any]) -> str:
    """Process user data with proper exception chaining.
    
    Args:
        user_data: Dictionary containing user information
        
    Returns:
        Success message with user email
        
    Raises:
        UserProcessingError: When processing fails (with chained original exception)
    """
    try:
        # Validate required fields
        email = user_data["email"]
    except KeyError as e:
        raise UserProcessingError("Missing required field: email") from e
    
    try:
        # Validate email format
        if "@" not in email:
            raise ValueError("Email must contain @ symbol")
    except ValueError as e:
        raise UserProcessingError(f"Invalid email format: {email}") from e
    
    try:
        # Validate email length
        if len(email) > 50:
            raise ValueError("Email too long")
    except ValueError as e:
        raise UserProcessingError(f"Email validation failed: {email}") from e
    
    return f"User {email} processed successfully"

# =============================================================================
# TASK 8: Context Manager for Timing Operations - SOLUTION
# =============================================================================

class TimerContext:
    """Context manager that times operations and provides timing information.
    
    Attributes:
        elapsed_time: Time elapsed during the context (set after __exit__)
        
    Example:
        >>> with TimerContext() as timer:
        ...     time.sleep(0.1)
        >>> timer.elapsed_time > 0.1
        True
    """
    
    def __init__(self):
        self.start_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        """Enter the context and start timing."""
        print("Starting operation...")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and calculate elapsed time."""
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time
        
        if exc_type is None:
            print(f"Operation completed in {self.elapsed_time:.3f} seconds")
        else:
            print(f"Operation failed after {self.elapsed_time:.3f} seconds")
        
        # Don't suppress exceptions
        return False

# =============================================================================
# TASK 9: Data Processing Pipeline with Comprehensive Error Handling - SOLUTION
# =============================================================================

class DataProcessingError(Exception):
    """Exception raised when data processing pipeline fails"""
    pass

def process_data_pipeline(config_file: str, data_file: str) -> str:
    """Process data according to configuration with comprehensive error handling.
    
    Args:
        config_file: Path to JSON configuration file
        data_file: Path to JSON data file
        
    Returns:
        Success message with processing details
        
    Raises:
        DataProcessingError: When any step of processing fails
    """
    try:
        # Read configuration
        with open(config_file) as f:
            config = json.load(f)
    except FileNotFoundError as e:
        raise DataProcessingError(f"Configuration file not found: {config_file}") from e
    except json.JSONDecodeError as e:
        raise DataProcessingError(f"Invalid JSON in config file: {config_file}") from e
    except Exception as e:
        raise DataProcessingError(f"Error reading config file: {config_file}") from e
    
    try:
        # Read data
        with open(data_file) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise DataProcessingError(f"Data file not found: {data_file}") from e
    except json.JSONDecodeError as e:
        raise DataProcessingError(f"Invalid JSON in data file: {data_file}") from e
    except Exception as e:
        raise DataProcessingError(f"Error reading data file: {data_file}") from e
    
    try:
        # Validate configuration
        if "multiplier" not in config:
            raise ValueError("Missing 'multiplier' in configuration")
        if "output_file" not in config:
            raise ValueError("Missing 'output_file' in configuration")
        
        multiplier = config["multiplier"]
        output_file = config["output_file"]
        
        # Process data
        if not isinstance(data, list):
            raise ValueError("Data must be a list of objects")
        
        results = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or "value" not in item:
                raise ValueError(f"Item {i} must be an object with 'value' field")
            results.append(item["value"] * multiplier)
        
        # Write results
        with open(output_file, 'w') as f:
            json.dump(results, f)
        
        return f"Processed {len(results)} items, results written to {output_file}"
        
    except ValueError as e:
        raise DataProcessingError(f"Data validation error: {e}") from e
    except Exception as e:
        raise DataProcessingError(f"Error processing data: {e}") from e

# =============================================================================
# BONUS TASK: Advanced Decorators - SOLUTION
# =============================================================================

def retry_with_backoff(max_attempts: int = 3, backoff_factor: float = 1.5):
    """Decorator that retries failed functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for wait time between retries
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise the exception
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f} seconds...")  # noqa : E501
                    time.sleep(wait_time)
            
            # This should never be reached, but just in case
            raise last_exception
        return wrapper
    return decorator

def log_calls(include_args: bool = True, include_result: bool = True):
    """Decorator that logs function calls with timestamps.
    
    Args:
        include_args: Whether to include arguments in log
        include_result: Whether to include result in log
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Log function call
            if include_args:
                args_str = ", ".join(repr(arg) for arg in args)
                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                print(f"[{timestamp}] Calling {func.__name__}({all_args})")
            else:
                print(f"[{timestamp}] Calling {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    print(f"[{timestamp}] {func.__name__} returned: {repr(result)}")
                else:
                    print(f"[{timestamp}] {func.__name__} completed successfully")
                
                return result
            except Exception as e:
                print(f"[{timestamp}] {func.__name__} raised exception: {e}")
                raise
        return wrapper
    return decorator

@retry_with_backoff(max_attempts=3, backoff_factor=2.0)
@log_calls(include_args=True, include_result=True)
def flaky_operation(operation_name: str) -> str:
    """Simulate an operation that fails randomly (30% success rate).
    
    Args:
        operation_name: Name of the operation being performed
        
    Returns:
        Success message
        
    Raises:
        ConnectionError: When the operation fails (70% of the time)
    """
    if random.random() < 0.3:  # 30% success rate
        return f"Operation '{operation_name}' completed successfully"
    else:
        raise ConnectionError(f"Failed to complete operation '{operation_name}'")

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_all_solutions():
    """Test all implemented solutions"""
    print("Testing all Functions and Exceptions solutions...\n")
    
    # Test Task 1
    print("Task 1 - Function Design:")
    print(f"Average of [1,2,3,-1,-2]: {calculate_positive_average([1, 2, 3, -1, -2])}")
    print(f"Empty list: {calculate_positive_average([])}")
    print(f"No positives: {calculate_positive_average([-1, -2, -3])}")
    print("✓ Task 1 complete\n")
    
    # Test Task 2
    print("Task 2 - User Creation:")
    user1 = create_user_profile("Alice", "alice@example.com")
    user2 = create_user_profile("Bob", "bob@example.com", age=25, department="Engineering")
    print(f"User 1: {user1['name']} - Active: {user1['active']}")
    print(f"User 2: {user2['name']} - Age: {user2.get('age')} - Dept: {user2.get('department')}")
    print("✓ Task 2 complete\n")
    
    # Test Task 3
    print("Task 3 - Flexible Calculator:")
    print(f"Add 1,2,3,4,5: {flexible_calculator(1, 2, 3, 4, 5)}")
    print(f"Multiply 2,3,4: {flexible_calculator(2, 3, 4, operation='multiply')}")
    print(f"Average 1.234,5.678 (rounded): {flexible_calculator(1.234, 5.678, operation='average', round_to=2)}")  # noqa : E501
    print("✓ Task 3 complete\n")
    
    # Test Task 4
    print("Task 4 - Validation Decorator:")
    print(process_age(25))
    print(process_email("test@example.com"))
    try:
        process_age(200)  # Should fail
    except ValueError as e:
        print(f"Validation caught: {e}")
    print("✓ Task 4 complete\n")
    
    # Test Task 5
    print("Task 5 - Banking Exceptions:")
    print(f"Successful withdrawal: {withdraw_money(1000, 500)}")
    try:
        withdraw_money(1000, 1500)
    except InsufficientFundsError as e:
        print(f"Caught: {e}")
    print("✓ Task 5 complete\n")

if __name__ == "__main__":
    test_all_solutions()