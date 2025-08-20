# Python Workshop
Python for Non-Python Develop## Setup Instructions

### Prerequisites
- Python 3.11+
- Basic understanding of Java, C#, C++, or JavaScript

### Environment Setup
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -e .
   ```

### Running the Workshop
- Work through each module in order, starting with `src/a_basics/`
- Each module contains both explanation files (.md) and practice exercises (.py)
- Solutions are provided in `xample.py` files within each module

### Testing Your Work
- Run basic validation: `python -m pytest -v` (when tests are implemented)
- Run individual modules: `python src/a_basics/a_variables.py`

## Learning Resources
- **docs/EXAMPLES.md** - Direct translations from Java/C#/C++ to Python
- **xample.py files** - Complete solutions and patterns for each module

## Tips for Java/C#/C++ Developers
- Focus on understanding Python's dynamic typing and duck typing
- Learn list/dict comprehensions - they're more powerful than loops
- Embrace Python's "batteries included" philosophy
- Don't try to write Java/C# in Python - learn the Pythonic waypository contains hands-on exercises for an intensive Python workshop aimed at experienced developers coming from languages like Java, C#, C++, and JavaScript.

## Workshop Structure

### Core Learning Path (Recommended Order)
1. **`src/a_basics/`** - Python Fundamentals
   - Variables, data types, operators, control flow
   - Essential for developers new to Python syntax

2. **`src/b_pythonic/`** - Pythonic Thinking  
   - Core Pythonic concepts, collections, comprehensions, unpacking, truthiness
   - Learn to "think in Python" vs other languages

3. **`src/c_func_exceptions/`** - Functions & Exception Handling
   - Functions, *args/**kwargs, closures, exceptions, context managers
   - Error handling patterns and functional programming concepts

4. **`src/d_modules_typing_tools/`** - Modules, Typing & Tools
   - Modules/packages, typing, dataclasses, Protocols, development tooling
   - Code organization and type safety

5. **`src/e_tests/`** - Testing in Python
   - pytest, testing patterns, mocking
   - Quality assurance practices

### Optional Advanced Topics
6. **`src/f_async/`** - Asynchronous Programming
   - async/await, asyncio, concurrent programming
   - Modern Python concurrency patterns

7. **`src/h_advanced/`** - Advanced Topics
   - Complex project structure patterns
   - Plugin architectures and advanced design patterns
   - For experienced developers ready for sophisticated Python patternsorkshop
Python for non-python programmers

This repository contains hands-on exercises for an intensive Python workshop aimed at experienced developers coming from languages like Java, C#, C++, and JavaScript.

Structure
- src/01_pythonic: Core Pythonic concepts, collections, comprehensions, unpacking, truthiness
- src/02_func_exceptions: Functions, *args/**kwargs, closures, exceptions, context managers
- src/03_modules_typing_tools: Modules/packages, typing, dataclasses, Protocols, tooling
- tests/: Pytest-based tests validating each section’s exercises

Setup
- Python 3.11+
- Create a virtual environment and install dev tools if desired

Running tests
- Run all tests: `pytest -q`
- Run a single file: `pytest -q tests/test_01_pythonic.py`

Examples list
- See docs/EXAMPLES.md for concrete “from Java/C#/C++ to Python” translations and idioms.

Tip: Use the panic.py in each module (when present) for hints and patterns.
