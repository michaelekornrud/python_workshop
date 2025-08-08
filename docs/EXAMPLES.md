# From Java/C#/C++/JS to Python — practical translations and idioms

Note: All examples target Python 3.11+.

Variables, types, and functions
- Dynamic by default, optional type hints for tooling

```python
x: int = 42  # hint, not enforced at runtime

def greet(name: str = "world") -> str:
    return f"Hello, {name}!"
```

Data classes vs POJOs/DTOs
- Java/C#: verbose class with getters/setters
- Python: `@dataclass` for value objects

```python
from dataclasses import dataclass

@dataclass(slots=True)
class User:
    id: int
    name: str
    email: str | None = None

u = User(1, "Ada")
```

Properties vs getters/setters
- Prefer `@property`

```python
class Account:
    def __init__(self, balance: float) -> None:
        self._balance = balance

    @property
    def balance(self) -> float:
        return self._balance

    @balance.setter
    def balance(self, value: float) -> None:
        if value < 0:
            raise ValueError("negative balance")
        self._balance = value
```

Interfaces and abstractions
- Java/C#: interface, C++: abstract class, C#: interface
- Python: `abc.ABC` or `typing.Protocol` (duck typing-friendly)

```python
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def get(self, id: int) -> object: ...

class InMemoryRepo(Repository):
    def __init__(self):
        self._data = {}
    def get(self, id: int) -> object:
        return self._data.get(id)
```

```python
from typing import Protocol

class SupportsLen(Protocol):
    def __len__(self) -> int: ...

def is_empty(x: SupportsLen) -> bool:
    return len(x) == 0
```

Overloading vs default/keyword arguments
- Python has no traditional overloads; use defaults/keywords and single-dispatch

```python
from functools import singledispatch

@singledispatch
def to_str(x) -> str: return str(x)

@to_str.register
def _(x: bytes) -> str: return x.decode()

# or defaults/keywords

def connect(host: str, *, timeout: float = 3.0, ssl: bool = True):
    ...
```

Collections and iteration
- Prefer iteration helpers over indices

```python
nums = [10, 20, 30]

for i, v in enumerate(nums):
    print(i, v)

letters = ["a", "b", "c"]
for n, L in zip(nums, letters):
    print(n, L)

squares = [n * n for n in nums]             # list comprehension
mapping = {c: i for i, c in enumerate(letters)}  # dict comp
unique = {x % 3 for x in range(10)}         # set comp
```

Truthiness and None

```python
if not []:  # empty containers are falsy
    ...

x = None
if x is None:  # identity for None
    ...
```

Pattern matching vs switch

```python
match obj:
    case {"type": "user", "id": int(id)}:
        ...
    case [x, y]:
        ...
    case _:
        ...
```

Exceptions over error codes

```python
try:
    risky()
except (IOError, ValueError) as e:
    raise RuntimeError("wrap") from e
```

Context managers vs try/finally

```python
# before
f = open("data.txt")
try:
    data = f.read()
finally:
    f.close()

# pythonic
from pathlib import Path
with Path("data.txt").open() as f:
    data = f.read()
```

Generators and lazy iteration

```python
def count_up(n: int):
    i = 0
    while i < n:
        yield i
        i += 1

for x in count_up(3):
    print(x)
```

Async basics (akin to JS async/await)

```python
import asyncio

async def fetch(url: str) -> str:
    await asyncio.sleep(0.1)
    return f"<html from {url}>"

async def main():
    html1, html2 = await asyncio.gather(fetch("/a"), fetch("/b"))

asyncio.run(main())
```

Equality, hashing, ordering

```python
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int

p = {Point(1, 2)}  # hashable because frozen
```

File paths and I/O

```python
from pathlib import Path
p = Path("/tmp") / "file.txt"
if p.exists():
    data = p.read_text(encoding="utf-8")
```

Testing with pytest

```python
# tests/test_example.py
import pytest
from example import add

def test_add():
    assert add(2, 3) == 5
```

Small CLI with argparse and typer

```python
# argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()
print(f"Hello {args.name}")
```

```python
# typer (pip install typer[all])
import typer

app = typer.Typer()

@app.command()
def hello(name: str = "world"):
    print(f"Hello {name}")

if __name__ == "__main__":
    app()
```

Common gotchas
- Mutable default args: use None sentinel

```python
def append_item(x=None):
    x = [] if x is None else x
    x.append(1)
    return x
```
- Late binding in closures: capture with default

```python
funcs = [lambda i=i: i for i in range(3)]
```
- Copy vs deepcopy

```python
import copy
shallow = copy.copy(obj)
deep = copy.deepcopy(obj)
```
