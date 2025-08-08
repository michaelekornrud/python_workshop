# ptyhon_workshop
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
