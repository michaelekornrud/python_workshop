# Modules in Python

# __init__.py file is used to mark a directory as a Python package. Found in src/00_basics/__init__.py

# Importing Modules
import os  # Importing the os module for operating system functionalities
import sys  # Importing the sys module for system-specific parameters and functions

# Using Functions from Imported Modules
print(os.getcwd())  # Prints the current working directory
print(sys.version)  # Prints the Python version


# Importing Specific Functions from a Module
from math import sqrt, pi  # Importing specific functions from the math module

# Using Imported Functions
print(sqrt(16))  # Prints the square root of 16
print(pi)  # Prints the value of pi

# Import the k_functions module (changed from relative to absolute import)
print("\n\nBefore importing k_functions: \n")
import k_functions as functions
print("\nAfter importing k_functions \n\n")

def using_imported_functions():
    """Function to demonstrate usage of imported functions."""

    name = "Alice"
    greeting = functions.greet(name)  # Using the greet function from k_functions
    print(greeting)  # Prints the greeting message


# Example of importing with aliases (using built-in modules)
import datetime as dt  # Importing datetime with an alias

# Using the Aliased Module
current_time = dt.datetime.now()  # Creating a datetime object
print(current_time)  # Prints the current datetime


if __name__ == "__main__":
    # This block will only execute if this script is run directly, not when imported
    using_imported_functions()  # Call the function to demonstrate usage of imported functions
    print("This script demonstrates module imports and usage in Python.")

    # You can run this script to see the output of the imported functions and modules.
    # To run this script, use the command: python src/00_basics/11_modules.py

    # Things to consider:
    # 1. Since k_funtions' functions are all imported, you can use them directly.
    #    But since the functions are not placed in classes, they will be executed on import.
    

