"""
Functions and Exceptions Workshop - Practice Exercises

🎯 WORKSHOP PRIORITY GUIDE:
- ⭐ CORE TASKS (1-3): Essential for 3-hour workshop - focus on these first
- 🔥 RECOMMENDED TASKS (4-5): Complete if time allows in workshop
- 💡 EXTENSION TASKS (6-9): Practice after workshop for deeper learning

Complete the following tasks to practice function design and exception handling.
Apply the principles from the markdown file to create robust, well-designed functions.
"""

import json
import time  # noqa : F401
from functools import wraps  # noqa : F401
from typing import Any

# =============================================================================
# ⭐ CORE TASK 1: Basic Function Design (Essential - 10 minutes)
# =============================================================================

"""
⭐ CORE TASK 1: Calculate Average of Positive Numbers

Create a function that calculates the average of positive numbers in a list.

Requirements:
- Function name: calculate_positive_average
- Parameters: numbers (list of numbers)
- Return: float (average) or None if no positive numbers found
- Include proper docstring with Google style documentation
- Use type hints for parameters and return value
- Handle edge cases: empty list, no positive numbers
- Use Pythonic patterns (list comprehensions, built-in functions)

Example usage:
calculate_positive_average([1, 2, 3, -1, -2]) -> 2.0
calculate_positive_average([-1, -2, -3]) -> None
calculate_positive_average([]) -> None
"""

def calculate_positive_average(numbers: list[int | float]) -> float | None:
    """Your solution here"""
    pass

# =============================================================================
# ⭐ CORE TASK 2: User Creation with Flexible Parameters (Essential - 10 minutes)
# =============================================================================

"""
⭐ CORE TASK 2: Create User Profile Function

Create a function that builds user profiles with required and optional information.

Requirements:
- Function name: create_user_profile
- Required parameters: name (str), email (str)
- Optional parameters with defaults: age=None, phone=None, active=True
- Accept additional optional fields via **kwargs
- Always include a 'created_at' timestamp
- Return a dictionary with all user information
- Use proper type hints and docstring

Example usage:
create_user_profile("Alice", "alice@example.com")
create_user_profile("Bob", "bob@example.com", age=25, phone="123-456-7890")
create_user_profile("Charlie", "charlie@example.com", department="Engineering", role="Developer")
"""

def create_user_profile(name: str, email: str, age: int | None = None, 
                       phone: str | None = None, active: bool = True, **kwargs) -> dict[str, Any]:
    """Your solution here"""
    pass

# =============================================================================
# ⭐ CORE TASK 3: Flexible Calculator with *args and **kwargs (Essential - 12 minutes)
# =============================================================================

"""
⭐ CORE TASK 3: Create a Flexible Calculator Function

Create a calculator function that can perform operations on any number of arguments.

Requirements:
- Function name: flexible_calculator
- Use *args to accept any number of numeric arguments
- Use **kwargs for operation type and options
- Support operations: "add" (default), "multiply", "average"
- Support options: round_to (integer for decimal places)
- Include proper error handling for invalid operations
- Return appropriate numeric result
- Use proper type hints and docstring

Example usage:
flexible_calculator(1, 2, 3, 4, 5) -> 15 (default add)
flexible_calculator(2, 3, 4, operation="multiply") -> 24
flexible_calculator(1.234, 5.678, operation="average", round_to=2) -> 3.46
"""

def flexible_calculator(*args, **kwargs) -> int | float:
    """Your solution here"""
    pass

# =============================================================================
# 🔥 RECOMMENDED TASK 4: Input Validation Decorator (If time allows - 15 minutes)
# =============================================================================

"""
🔥 RECOMMENDED TASK 4: Create Input Validation Decorator

Create a decorator that validates function inputs based on specified criteria.

Requirements:
- Function name: validate_input
- Parameters: input_type (str), min_val=None, max_val=None, required_char=None
- Should validate the first argument of decorated functions
- Support type validation ("int", "str", "float")
- Support range validation for numbers (min_val, max_val)
- Support character requirement for strings (required_char)
- Raise appropriate exceptions (TypeError, ValueError) with clear messages
- Use proper type hints and docstring

Example usage:
@validate_input("int", min_val=0, max_val=150)
def process_age(age):
    return f"Processing age: {age}"

@validate_input("str", required_char="@")
def process_email(email):
    return f"Processing email: {email}"
"""

def validate_input(input_type: str, min_val: int | float | None = None, 
                  max_val: int | float | None = None, required_char: str | None = None):
    """Your decorator solution here"""
    pass

# Example functions to test your decorator
@validate_input("int", min_val=0, max_val=150)
def process_age(age: int) -> str:
    """Process age with decorator validation"""
    return f"Processing age: {age}"

@validate_input("str", required_char="@")
def process_email(email: str) -> str:
    """Process email with decorator validation"""
    return f"Processing email: {email}"

# =============================================================================
# 🔥 RECOMMENDED TASK 5: Custom Exception Classes for Banking (If time allows - 15 minutes)
# =============================================================================

"""
🔥 RECOMMENDED TASK 5: Create Banking Exception Hierarchy

Create a banking system with custom exceptions for different error conditions.

Requirements:
- Base exception class: BankError (inherits from Exception)
- Specific exceptions: InsufficientFundsError, InvalidAmountError, DailyLimitError
- Each exception should accept and store relevant information (amount, balance, limit)
- Create function: withdraw_money(balance, amount, daily_limit=10000)
- Function should validate inputs and raise appropriate custom exceptions
- Include proper docstrings and type hints

Example usage:
withdraw_money(1000, 500) -> 500 (new balance)
withdraw_money(1000, 1500) -> raises InsufficientFundsError
withdraw_money(1000, -100) -> raises InvalidAmountError
withdraw_money(1000, 15000) -> raises DailyLimitError
"""

class BankError(Exception):
    """Base exception for banking operations"""
    pass

class InsufficientFundsError(BankError):
    """Your custom exception here"""
    pass

class InvalidAmountError(BankError):
    """Your custom exception here"""
    pass

class DailyLimitError(BankError):
    """Your custom exception here"""
    pass

def withdraw_money(balance: float, amount: float, daily_limit: float = 10000) -> float:
    """Your solution here"""
    pass

# =============================================================================
# 💡 EXTENSION TASK 6: JSON File Processing with Error Handling (Post-workshop practice - 12 minutes)
# =============================================================================

"""
💡 EXTENSION TASK 6: Safe JSON File Reader

Create a function that safely reads and parses JSON files with comprehensive error handling.

Requirements:
- Function name: read_json_file
- Parameter: filename (str)
- Return: parsed JSON data or raise descriptive exceptions
- Use context managers for file operations
- Handle specific exceptions: FileNotFoundError, PermissionError, json.JSONDecodeError
- Raise custom exceptions with meaningful error messages
- Include proper type hints and docstring

Example usage:
read_json_file("data.json") -> parsed data
read_json_file("missing.json") -> raises FileNotFoundError with context
read_json_file("invalid.json") -> raises ValueError with JSON error details
"""

def read_json_file(filename: str) -> dict[str, Any] | list[Any]:
    """Your solution here"""
    pass

# =============================================================================
# 💡 EXTENSION TASK 7: User Data Processing with Exception Chaining (Post-workshop - 12 min)
# =============================================================================

"""
💡 EXTENSION TASK 7: User Data Processor with Exception Chaining

Create a function that processes user data with proper exception chaining.

Requirements:
- Function name: process_user_data
- Parameter: user_data (dict)
- Validate required fields (email)
- Validate email format (contains @)
- Validate email length (max 50 characters)
- Use exception chaining (raise ... from ...) to preserve original error context
- Create custom exception: UserProcessingError
- Include proper type hints and docstring

Example usage:
process_user_data({"email": "test@example.com"}) -> "User test@example.com processed"
process_user_data({}) -> raises UserProcessingError with chained KeyError
process_user_data({"email": "invalid"}) -> raises UserProcessingError with chained ValueError
"""

class UserProcessingError(Exception):
    """Your custom exception here"""
    pass

def process_user_data(user_data: dict[str, Any]) -> str:
    """Your solution here"""
    pass

# =============================================================================
# TASK 8: Context Manager for Timing Operations
# =============================================================================

"""
TASK 8: Timer Context Manager

Create a context manager that times operations and provides timing information.

Requirements:
- Class name: TimerContext
- Should implement __enter__ and __exit__ methods
- Print start message when entering context
- Print completion message with elapsed time when exiting
- Store elapsed time in a 'elapsed_time' attribute
- Handle exceptions gracefully (don't suppress them)
- Include proper type hints and docstring

Example usage:
with TimerContext() as timer:
    time.sleep(0.1)  # Simulate work
print(f"Operation took {timer.elapsed_time:.3f} seconds")
"""

class TimerContext:
    """Your context manager solution here"""
    pass

# =============================================================================
# TASK 9: Data Processing Pipeline with Comprehensive Error Handling
# =============================================================================

"""
TASK 9: Data Processing Pipeline

Create a comprehensive data processing function that handles multiple file operations.

Requirements:
- Function name: process_data_pipeline
- Parameters: config_file (str), data_file (str)
- Read configuration from config_file (JSON)
- Read data from data_file (JSON)
- Process data according to config (multiply values by config multiplier)
- Write results to output file specified in config
- Handle all possible exceptions with specific error messages
- Use context managers for all file operations
- Create custom exception: DataProcessingError
- Include proper cleanup and validation

Example config.json:
{"multiplier": 2.5, "output_file": "results.json"}

Example data.json:
[{"value": 10}, {"value": 20}, {"value": 30}]

Expected result: [25.0, 50.0, 75.0] written to results.json
"""

class DataProcessingError(Exception):
    """Your custom exception here"""
    pass

def process_data_pipeline(config_file: str, data_file: str) -> str:
    """Your solution here"""
    pass

# =============================================================================
# BONUS TASK: Advanced Decorators
# =============================================================================

"""
BONUS TASK: Create Advanced Function Decorators

Create two sophisticated decorators that can be combined.

Requirements:

1. retry_with_backoff decorator:
   - Parameters: max_attempts=3, backoff_factor=1.5
   - Retry failed functions with exponential backoff
   - Wait backoff_factor^attempt seconds between retries
   - Re-raise the last exception after max_attempts

2. log_calls decorator:
   - Parameters: include_args=True, include_result=True
   - Log function calls with timestamps
   - Optionally include arguments and results
   - Log exceptions when they occur

3. flaky_operation function:
   - Simulate an operation that fails randomly
   - Should succeed approximately 30% of the time
   - When it fails, raise a ConnectionError

Example usage:
@retry_with_backoff(max_attempts=3, backoff_factor=2.0)
@log_calls(include_args=True, include_result=True)
def flaky_operation(operation_name):
    # Implementation that sometimes fails
"""

def retry_with_backoff(max_attempts: int = 3, backoff_factor: float = 1.5):
    """Your retry decorator here"""
    pass

def log_calls(include_args: bool = True, include_result: bool = True):
    """Your logging decorator here"""
    pass

@retry_with_backoff(max_attempts=3, backoff_factor=2.0)
@log_calls(include_args=True, include_result=True)
def flaky_operation(operation_name: str) -> str:
    """Your flaky operation here"""
    pass

# =============================================================================
# TEST FUNCTIONS - Run these to check your solutions
# =============================================================================

def test_all_tasks():
    """Test all implemented functions"""
    print("Testing Functions and Exceptions implementations...\n")
    
    # Test Task 1
    print("Task 1 - Function Design:")
    try:
        result = calculate_positive_average([1, 2, 3, 4, 5, -1, -2])
        print(f"Average of positive numbers: {result}")
        print(f"Empty list handling: {calculate_positive_average([])}")
        print(f"No positive numbers: {calculate_positive_average([-1, -2, -3])}")
        print("✓ Task 1 working\n")
    except Exception as e:
        print(f"✗ Task 1 error: {e}\n")
    
    # Test Task 2
    print("Task 2 - Function Arguments:")
    try:
        user1 = create_user_profile("Alice", "alice@example.com")
        user2 = create_user_profile("Bob", "bob@example.com", age=25, phone="123-456-7890")
        print(f"Basic user: {user1.get('name')} - Active: {user1.get('active')}")
        print(f"Extended user: {user2.get('name')} - Age: {user2.get('age')}")
        print("✓ Task 2 working\n")
    except Exception as e:
        print(f"✗ Task 2 error: {e}\n")
    
    # Test Task 3
    print("Task 3 - *args and **kwargs:")
    try:
        result1 = flexible_calculator(1, 2, 3, 4, 5)
        result2 = flexible_calculator(2, 3, 4, operation="multiply")
        result3 = flexible_calculator(1.234, 5.678, operation="average", round_to=2)
        print(f"Add: {result1}, Multiply: {result2}, Rounded: {result3}")
        print("✓ Task 3 working\n")
    except Exception as e:
        print(f"✗ Task 3 error: {e}\n")
    
    # Test Task 4
    print("Task 4 - Decorators:")
    try:
        result1 = process_age(25)
        result2 = process_email("test@example.com")
        print(f"Age result: {result1}")
        print(f"Email result: {result2}")
        print("✓ Task 4 working\n")
    except Exception as e:
        print(f"✗ Task 4 error: {e}\n")
    
    # Test Task 5
    print("Task 5 - Custom Exceptions:")
    try:
        result = withdraw_money(1000, 500)
        print(f"Withdrawal successful: {result}")
        
        # Test exceptions
        try:
            withdraw_money(1000, 1500)
        except InsufficientFundsError as e:
            print(f"Caught insufficient funds: {e}")
        
        print("✓ Task 5 working\n")
    except Exception as e:
        print(f"✗ Task 5 error: {e}\n")
    
    # Test Task 6
    print("Task 6 - Exception Handling:")
    try:
        # Create a test JSON file
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        with open("test.json", "w") as f:
            json.dump(test_data, f)
        
        result = read_json_file("test.json")
        print(f"JSON read successfully: {result}")
        print("✓ Task 6 working\n")
        
        # Clean up
        import os
        os.remove("test.json")
    except Exception as e:
        print(f"✗ Task 6 error: {e}\n")

if __name__ == "__main__":
    test_all_tasks()
