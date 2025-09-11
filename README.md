# Python Workshop
Python for Non-Python Developers

## Setup Instructions

### Prerequisites
- Python 3.11+ (required by project dependencies)
- UV package manager
- Git (for cloning the repository)
- A code editor/IDE (VS Code, PyCharm, or similar recommended)
- Basic understanding of Java, C#, C++, or JavaScript

#### Development Dependencies
This workshop includes several Python development tools:
- **pytest** - Testing framework
- **ruff** - Fast Python linter and formatter
- **pandas** - Data manipulation library (used in advanced examples)
- **aiohttp & aiofiles** - Async HTTP and file operations (for async module)

#### Python Installation

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
Visit https://www.python.org/downloads/macos/
```

**Windows:**
```powershell
# Using winget (Windows 10+)
winget install Python.Python.3.12

# Or using Chocolatey
choco install python

# Or download from python.org
Visit https://www.python.org/downloads/windows/
IMPORTANT: When using the installer, check "Add Python to PATH" during installation
```

**Note for Windows users:** If you installed Python but `python --version` doesn't work in your terminal:
1. **For manual python.org installation:** Re-run the installer and check "Add Python to PATH"
2. **Or manually add to PATH:**
   - Search for "Environment Variables" in Windows Start menu
   - Click "Environment Variables" → "Path" → "Edit"
   - Add Python installation directory (usually `C:\Users\<username>\AppData\Local\Programs\Python\Python312\`)
   - Add Scripts directory (usually `C:\Users\<username>\AppData\Local\Programs\Python\Python312\Scripts\`)
   - Restart your terminal

#### UV Installation

**macOS:**
```bash
# Using Homebrew (recommended)
brew install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
# Using winget
winget install astral-sh.uv

# Or using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using Scoop
scoop install uv
```

### Environment Setup
1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd ptyhon_workshop
   ```

2. **Verify Python version:**
   ```bash
   python --version  # Should show 3.11 or higher
   ```

3. **Install dependencies using UV:**
   ```bash
   uv sync
   ```

4. **Verify installation:**
   ```bash
   # Test that dependencies are installed correctly
   uv run python -c "import pytest, pandas, aiohttp; print('All dependencies installed successfully!')"
   ```

5. **Run the test suite to ensure everything works:**
   ```bash
   uv run pytest tests/ -v
   ```

6. **Optional: Activate the virtual environment manually (if preferred):**
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   ```

#### IDE/Editor Setup Recommendations
- **VS Code**: Install the Python extension for syntax highlighting, debugging, and IntelliSense
- **PyCharm**: Comes with excellent Python support out of the box
- **Vim/Neovim**: Consider python-lsp-server or similar language server
- Configure your editor to use the virtual environment at `.venv/`

### Running the Workshop
- Work through each module in order, starting with `src/a_basics/`
- Each module contains both explanation files (.md) and practice exercises (.py)
- Solutions are provided in `xample.py` files within each module

### Testing Your Work
- Run basic validation: `uv run pytest -v` (when tests are implemented)
- Run individual modules: `uv run python src/a_basics/a_variables.py`

## Learning Resources
- **docs/EXAMPLES.md** - Direct translations from Java/C#/C++ to Python
- **xample.py files** - Complete solutions and patterns for each module

## Tips for Java/C#/C++ Developers
- Focus on understanding Python's dynamic typing and duck typing
- Learn list/dict comprehensions - they're more powerful than loops
- Embrace Python's "batteries included" philosophy
- Don't try to write Java/C# in Python - learn the Pythonic way

## Workshop Structure

### Core Learning Path (Recommended Order)
1. **`src/a_basics/`** - Python Fundamentals
   - Variables, data types, operators, control flow
   - Essential for developers new to Python syntax
   
2. **`src/b_func_exceptions/`** - Functions & Exception Handling
   - Functions, *args/**kwargs, closures, exceptions, context managers
   - Error handling patterns and functional programming concepts

3. **`src/c_pythonic/`** - Pythonic Thinking  
   - Core Pythonic concepts, collections, comprehensions, unpacking, truthiness
   - Learn to "think in Python" vs other languages

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

## Why UV?

This project uses UV instead of traditional pip/venv because:
- **Faster dependency resolution** - UV resolves dependencies significantly faster than pip
- **Better lock file management** - Provides reliable, reproducible builds
- **Simplified workflow** - No need to manually manage virtual environments
- **Modern Python tooling** - Built for the current Python ecosystem

## Running Tests

Run all tests:
```bash
uv run pytest -v
```

Run tests for a specific module:
```bash
uv run pytest tests/test_basics.py -v
```
