# Funtions in Python

# Function Definition
# Functions are defined using the `def` keyword followed by the function name and parentheses.
def greet(name):
    """Function to greet a person."""
    return f"Hello, {name}!"    

# Function Call
# To execute a function, you call it by its name followed by parentheses.
print(greet("Alice"))  # Output: Hello, Alice!

# Function with Parameters
# Functions can take parameters, which are specified in the parentheses.
def add(a, b):
    """Function to add two numbers."""
    return a + b

# Function with Default Parameters
# You can provide default values for parameters.
def multiply(a, b=1):
    """Function to multiply two numbers with a default value for b."""
    return a * b

# Function with Variable Number of Arguments
# You can use `*args` to accept a variable number of positional arguments.
def sum_all(*args):
    """Function to sum all provided arguments."""
    return sum(args)  

# Function with Keyword Arguments
# You can use `**kwargs` to accept a variable number of keyword arguments.
def print_info(**kwargs):
    """Function to print key-value pairs."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Lambda Functions
# Lambda functions are small anonymous functions defined with the `lambda` keyword.
square = lambda x: x ** 2  # Lambda function to square a number
print(square(5))  # Output: 25  

# Function Annotations
# You can annotate function parameters and return types for better readability.
def divide(a: float, b: float) -> float:
    """Function to divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Nested Functions
# Functions can be defined inside other functions.
def outer_function(x):
    """Outer function that defines an inner function."""
    def inner_function(y):
        """Inner function that adds x and y."""
        return x + y
    return inner_function   

# Function Decorators
# Decorators are functions that modify the behavior of another function.
def decorator_function(func):
    """Decorator function that adds functionality to another function."""
    def wrapper(*args, **kwargs):
        print("Before calling the function")
        result = func(*args, **kwargs)
        print("After calling the function")
        return result
    return wrapper

@decorator_function
def say_hello(name):
    """Function to say hello."""
    return f"Hello, {name}!"

print(say_hello("Bob"))  # Output: Before calling the function, Hello, Bob!, After calling the function

# Recursive Functions
# A recursive function is a function that calls itself.
def factorial(n):
    """Function to calculate the factorial of a number."""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1) 

print(factorial(5))  # Output: 120

# Function Scope
# Variables defined inside a function are local to that function.
def local_variable_example():
    """Function demonstrating local variable scope."""
    local_var = "I am local"
    print(local_var)  # This will work
    # print(global_var)  # This will raise an error if global_var is not defined

# Global Variables
# You can define variables outside of functions, which are accessible globally.
global_var = "I am global"

def global_variable_example():
    """Function demonstrating global variable scope."""
    print(global_var)  # This will work

# Function Returning Multiple Values
# Functions can return multiple values as a tuple.
def min_max(numbers):
    """Function to return the minimum and maximum of a list of numbers."""
    return min(numbers), max(numbers)

# Function Returning a Dictionary
# Functions can also return a dictionary.
def person_info(name, age):
    """Function to return a dictionary with person's info."""
    return {"name": name, "age": age}

# Function with Type Hints
# Type hints can be used to indicate the expected types of parameters and return values.    
def concatenate_strings(a: str, b: str) -> str:
    """Function to concatenate two strings."""
    return a + b

# Function with Optional Parameters
# You can use `Optional` from the `typing` module to indicate that a parameter can be `None`.
from typing import Optional # This is preffered up to Python 3.9
def greet_optional(name: str | None = None) -> str:
    """Function to greet a person, with an optional name."""
    if name is None:
        return "Hello, stranger!"
    return f"Hello, {name}!"

# From Python 3.10, you can use the `|` operator for union types.
# This provide a more concise way to specify optional types.
def greet_union(name: str | None = None) -> str:
    """Function to greet a person, with an optional name."""
    if name is None:
        return "Hello, stranger!"
    return f"Hello, {name}!"

# Function with Keyword-Only Arguments
# You can specify that certain parameters must be passed as keyword arguments.
def keyword_only_function(*, required_param: str, optional_param: str = "default"):
    """Function with keyword-only parameters."""
    return f"Required: {required_param}, Optional: {optional_param}"

# Function with Positional-Only Arguments
# You can specify that certain parameters must be passed as positional arguments.
def positional_only_function(a, b, /, c):
    """Function with positional-only parameters."""
    return a + b + c

# Function with Type Aliases
# You can create type aliases for better readability.
from typing import List, Tuple
Vector = list[float]  # Type alias for a list of floats
def vector_add(v1: Vector, v2: Vector) -> Vector:
    """Function to add two vectors."""
    return [x + y for x, y in zip(v1, v2)]

# Function with Context Managers
# Context managers are used to manage resources, like files or network connections.
from contextlib import contextmanager
@contextmanager
def managed_resource():
    """Context manager for managing a resource."""
    resource = "Resource acquired"
    try:
        yield resource  # Yield the resource to the block of code using it
    finally:
        print("Resource released")

with managed_resource() as res:
    print(res)  # Output: Resource acquired

# Function with Annotations for Documentation
# You can use annotations to provide additional information about the function.
def annotated_function(param1: int, param2: str) -> bool:
    """Function that takes an integer and a string, returning a boolean."""
    return str(param1) in param2

# Function with Docstrings
# Docstrings provide documentation for the function and can be accessed via the `__doc__`
print(annotated_function.__doc__)  # Output: Function that takes an integer and a string, returning a boolean.

# Function with Type Checking
# You can use the `isinstance` function to check the type of parameters.
def type_checked_function(param: int):
    """Function that checks if the parameter is an integer."""
    if not isinstance(param, int):
        raise TypeError("Parameter must be an integer")
    return param * 2

# Function with Assertions
# Assertions can be used to check conditions within a function.
def assert_function(param: int):
    """Function that asserts the parameter is a positive integer."""
    assert param > 0, "Parameter must be a positive integer"
    return param ** 2

# Function with Logging
# You can use the `logging` module to log messages from within a function.
import logging
logging.basicConfig(level=logging.INFO)
def logged_function(param: str):
    """Function that logs a message."""
    logging.info(f"Function called with parameter: {param}")
    return f"Logged: {param}"

# Function with Type Guards
# Type guards can be used to narrow down types within a function.
from typing import Union, TypeGuard
def is_str_list(value: str | list) -> TypeGuard[list]:
    """Type guard to check if value is a list of strings."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)

# Function with Type Variables
# Type variables can be used to create generic functions.
from typing import TypeVar, Generic
T = TypeVar('T')
def identity(value: T) -> T:
    """Generic function that returns the input value."""
    return value

# Function with Asyncio
# Asynchronous functions can be defined using the `async def` syntax.
import asyncio
async def async_function():
    """Asynchronous function that simulates a delay."""
    await asyncio.sleep(1)
    return "Async result"

# Function with Await
async def main():
    """Main function to run the async function."""
    result = await async_function()
    print(result)  # Output: Async result


if __name__ == "__main__":
    # This block will only execute if this script is run directly, not when imported
    asyncio.run(main())  # Run the main function
    print("This script demonstrates various types of functions in Python.")
    # You can run this script to see the output of the defined functions.
    # To run this script, use the command: python src/a_basics/k_functions.py

# Function with Async Generator (Coroutine)
from typing import AsyncGenerator
async def coroutine_function() -> AsyncGenerator[str, None]:
    """Async generator function that yields values."""
    yield "Coroutine started"
    yield "Coroutine ended"



async def run_coroutine():
    async for value in coroutine_function():
        print(value)


asyncio.run(run_coroutine())

# Function with Generators
# Generators can be defined using the `yield` keyword.
def generator_function(n: int):
    """Generator function that yields numbers from 0 to n."""
    yield from range(n)

def run_generator():
    """Function to run the generator and print its values."""
    for value in generator_function(5):
        print(value)  # Output: 0, 1, 2, 3, 4

run_generator()  # Run the generator function

# Function with Iterators
class MyIterator:
    """Custom iterator class."""
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            value = self.data[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration