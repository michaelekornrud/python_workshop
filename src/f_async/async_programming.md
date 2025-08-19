# Async Programming in Python

Asynchronous programming is a programming paradigm that enables concurrent execution of code without using multiple threads. In Python, async programming allows you to write code that can handle many operations simultaneously, making it particularly powerful for I/O-bound tasks like web requests, file operations, and database queries.

## Why Async Programming Matters

Traditional synchronous programming executes code line by line, waiting for each operation to complete before moving to the next. This is inefficient when dealing with operations that involve waiting - like network requests or file I/O. Async programming allows your program to do other work while waiting for these operations to complete.

## 1. Async/Await Fundamentals

The foundation of Python's async programming is built on `async` and `await` keywords, along with the `asyncio` library.

### Understanding Coroutines

A coroutine is a function that can be paused and resumed. In Python, you create coroutines using the `async def` syntax:

```python
import asyncio
import aiohttp
import time
from typing import List, Dict, Any

async def fetch_data(url: str) -> str:
    """Fetch data from a URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def simple_coroutine():
    """A simple coroutine that demonstrates async/await."""
    print("Starting coroutine")
    await asyncio.sleep(1)  # Simulate async work
    print("Coroutine completed")
    return "Result"

# Running coroutines
async def main():
    result = await simple_coroutine()
    print(f"Got result: {result}")

# Execute the async function
if __name__ == "__main__":
    asyncio.run(main())
```

### The Event Loop

The event loop is the heart of async programming. It manages and executes coroutines, handles I/O operations, and schedules callbacks:

```python
import asyncio

async def task_example(name: str, delay: float):
    """Example task that simulates work."""
    print(f"Task {name} starting")
    await asyncio.sleep(delay)
    print(f"Task {name} completed after {delay}s")
    return f"Result from {name}"

async def run_tasks():
    """Run multiple tasks concurrently."""
    # Create tasks
    task1 = asyncio.create_task(task_example("A", 2))
    task2 = asyncio.create_task(task_example("B", 1))
    task3 = asyncio.create_task(task_example("C", 3))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(task1, task2, task3)
    return results
```

## 2. Concurrent Execution Patterns

Async programming shines when you need to handle multiple operations concurrently. Here are the key patterns:

### asyncio.gather() - Running Tasks in Parallel

```python
async def fetch_multiple_urls(urls: List[str]) -> List[str]:
    """Fetch multiple URLs concurrently."""
    async def fetch_single(url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()
    
    # Fetch all URLs concurrently
    results = await asyncio.gather(*[fetch_single(url) for url in urls])
    return results

# Usage
urls = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/2",
    "https://httpbin.org/delay/1"
]
```

### asyncio.as_completed() - Processing Results as They Come

```python
async def process_as_completed(urls: List[str]):
    """Process URLs as they complete, not waiting for all."""
    async def fetch_with_info(url: str, index: int):
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.text()
                duration = time.time() - start_time
                return {
                    "url": url,
                    "index": index,
                    "duration": duration,
                    "size": len(content)
                }
    
    tasks = [fetch_with_info(url, i) for i, url in enumerate(urls)]
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Completed {result['url']} in {result['duration']:.2f}s")
        yield result
```

### Task Management and Cancellation

```python
async def cancellable_task(name: str, duration: float):
    """A task that can be cancelled."""
    try:
        print(f"Starting task {name}")
        await asyncio.sleep(duration)
        print(f"Task {name} completed")
        return f"Success: {name}"
    except asyncio.CancelledError:
        print(f"Task {name} was cancelled")
        raise

async def task_with_timeout():
    """Demonstrate task cancellation and timeouts."""
    task = asyncio.create_task(cancellable_task("long-running", 10))
    
    try:
        # Wait for task with timeout
        result = await asyncio.wait_for(task, timeout=3.0)
        return result
    except asyncio.TimeoutError:
        print("Task timed out")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print("Task cancelled successfully")
        return None
```

## 3. Async Context Managers and Iterators

### Async Context Managers

Async context managers help manage resources in async code:

```python
class AsyncDatabaseConnection:
    """Example async context manager for database connections."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    async def __aenter__(self):
        """Async enter method."""
        print("Connecting to database...")
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connection = f"Connected to {self.connection_string}"
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit method."""
        print("Closing database connection...")
        await asyncio.sleep(0.1)  # Simulate cleanup time
        self.connection = None
    
    async def execute_query(self, query: str):
        """Execute a database query."""
        if not self.connection:
            raise RuntimeError("Not connected to database")
        await asyncio.sleep(0.2)  # Simulate query time
        return f"Query result for: {query}"

# Usage
async def database_example():
    async with AsyncDatabaseConnection("postgresql://localhost:5432/mydb") as db:
        result = await db.execute_query("SELECT * FROM users")
        print(result)
```

### Async Iterators

Async iterators allow you to iterate over data that's produced asynchronously:

```python
class AsyncDataStream:
    """Async iterator that produces data over time."""
    
    def __init__(self, count: int, delay: float = 0.5):
        self.count = count
        self.delay = delay
        self.current = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current >= self.count:
            raise StopAsyncIteration
        
        await asyncio.sleep(self.delay)
        value = f"Data item {self.current}"
        self.current += 1
        return value

# Usage with async for
async def process_stream():
    async for item in AsyncDataStream(5, 0.2):
        print(f"Processing: {item}")
```

## 4. Error Handling in Async Code

Error handling in async code requires special consideration:

```python
async def risky_operation(should_fail: bool = False):
    """Operation that might fail."""
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError("Something went wrong!")
    return "Success"

async def handle_async_errors():
    """Demonstrate error handling patterns."""
    try:
        result = await risky_operation(should_fail=True)
        print(result)
    except ValueError as e:
        print(f"Caught error: {e}")
    
    # Handling errors in gather
    tasks = [
        risky_operation(False),
        risky_operation(True),
        risky_operation(False)
    ]
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {i} failed: {result}")
            else:
                print(f"Task {i} succeeded: {result}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## 5. Real-World Async Patterns

### Web API Client

```python
import aiohttp
import asyncio
from typing import List, Dict, Optional

class AsyncAPIClient:
    """Async HTTP client for web APIs."""
    
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make GET request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()
    
    async def post(self, endpoint: str, data: Dict = None) -> Dict:
        """Make POST request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def fetch_multiple(self, endpoints: List[str]) -> List[Dict]:
        """Fetch multiple endpoints concurrently."""
        tasks = [self.get(endpoint) for endpoint in endpoints]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Usage
async def api_example():
    async with AsyncAPIClient("https://jsonplaceholder.typicode.com") as client:
        # Single request
        post = await client.get("/posts/1")
        print(f"Post title: {post['title']}")
        
        # Multiple concurrent requests
        endpoints = ["/posts/1", "/posts/2", "/posts/3"]
        posts = await client.fetch_multiple(endpoints)
        
        for i, post in enumerate(posts):
            if isinstance(post, Exception):
                print(f"Request {i} failed: {post}")
            else:
                print(f"Post {i}: {post['title']}")
```

### Producer-Consumer Pattern with Queues

```python
import asyncio
import random
from asyncio import Queue
from typing import Any

class AsyncProducerConsumer:
    """Producer-consumer pattern using async queues."""
    
    def __init__(self, queue_size: int = 10):
        self.queue: Queue = Queue(maxsize=queue_size)
        self.running = False
    
    async def producer(self, name: str, items: List[Any]):
        """Producer that adds items to the queue."""
        print(f"Producer {name} starting")
        
        for item in items:
            await self.queue.put(item)
            print(f"Producer {name} added: {item}")
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        print(f"Producer {name} finished")
    
    async def consumer(self, name: str, process_time: float = 0.2):
        """Consumer that processes items from the queue."""
        print(f"Consumer {name} starting")
        
        while self.running:
            try:
                # Wait for item with timeout
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Process item
                print(f"Consumer {name} processing: {item}")
                await asyncio.sleep(process_time)
                
                # Mark task as done
                self.queue.task_done()
                print(f"Consumer {name} completed: {item}")
                
            except asyncio.TimeoutError:
                print(f"Consumer {name} timeout - checking if should continue")
                continue
        
        print(f"Consumer {name} stopping")
    
    async def run_system(self, producer_data: Dict[str, List[Any]], 
                        consumer_names: List[str], duration: float = 5.0):
        """Run the producer-consumer system."""
        self.running = True
        
        # Start producers
        producer_tasks = [
            asyncio.create_task(self.producer(name, items))
            for name, items in producer_data.items()
        ]
        
        # Start consumers
        consumer_tasks = [
            asyncio.create_task(self.consumer(name))
            for name in consumer_names
        ]
        
        # Wait for producers to finish
        await asyncio.gather(*producer_tasks)
        
        # Wait for queue to be empty
        await self.queue.join()
        
        # Stop consumers
        self.running = False
        
        # Wait for consumers to stop
        await asyncio.gather(*consumer_tasks, return_exceptions=True)

# Usage
async def queue_example():
    system = AsyncProducerConsumer(queue_size=5)
    
    producer_data = {
        "Producer-1": ["item-1", "item-2", "item-3"],
        "Producer-2": ["task-A", "task-B", "task-C", "task-D"]
    }
    
    consumer_names = ["Consumer-1", "Consumer-2"]
    
    await system.run_system(producer_data, consumer_names)
```

## 6. Performance and Best Practices

### Measuring Async Performance

```python
import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def async_timer(operation_name: str) -> AsyncGenerator[None, None]:
    """Context manager to time async operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        print(f"{operation_name} took {duration:.2f} seconds")

async def benchmark_async_vs_sync():
    """Compare async vs sync performance."""
    
    async def async_task(delay: float):
        await asyncio.sleep(delay)
        return f"Task completed after {delay}s"
    
    def sync_task(delay: float):
        time.sleep(delay)
        return f"Task completed after {delay}s"
    
    delays = [0.5, 0.3, 0.8, 0.2, 0.6]
    
    # Async version
    async with async_timer("Async execution"):
        tasks = [async_task(delay) for delay in delays]
        results = await asyncio.gather(*tasks)
    
    # Sync version (for comparison)
    with timer("Sync execution"):
        results = [sync_task(delay) for delay in delays]

@asynccontextmanager
def timer(operation_name: str):
    """Sync context manager for timing."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        print(f"{operation_name} took {duration:.2f} seconds")
```

### Common Pitfalls and Solutions

```python
# DON'T: Blocking operations in async code
async def bad_example():
    # This blocks the event loop!
    time.sleep(1)  # Wrong!
    return "Done"

# DO: Use async alternatives
async def good_example():
    # This doesn't block the event loop
    await asyncio.sleep(1)  # Correct!
    return "Done"

# DON'T: Forgetting to await
async def bad_await_example():
    # This doesn't actually wait!
    asyncio.sleep(1)  # Wrong - missing await!
    return "Done"

# DO: Always await coroutines
async def good_await_example():
    # This properly waits
    await asyncio.sleep(1)  # Correct!
    return "Done"

# DON'T: Creating too many tasks without limits
async def bad_task_creation(urls: List[str]):
    # This could create thousands of tasks!
    tasks = [fetch_data(url) for url in urls]  # Potentially dangerous
    return await asyncio.gather(*tasks)

# DO: Use semaphores to limit concurrency
async def good_task_creation(urls: List[str], max_concurrent: int = 10):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_fetch(url: str):
        async with semaphore:
            return await fetch_data(url)
    
    tasks = [limited_fetch(url) for url in urls]
    return await asyncio.gather(*tasks)
```

## 7. Testing Async Code

Testing async code requires special considerations:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

# Using pytest-asyncio for testing
@pytest.mark.asyncio
async def test_async_function():
    """Test an async function."""
    result = await simple_coroutine()
    assert result == "Result"

@pytest.mark.asyncio
async def test_async_with_mock():
    """Test async function with mocking."""
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Create async mock
        mock_response = AsyncMock()
        mock_response.text.return_value = "Mocked response"
        mock_get.return_value.__aenter__.return_value = mock_response
        
        result = await fetch_data("http://example.com")
        assert "Mocked" in result

# Fixtures for async testing
@pytest.fixture
async def async_client():
    """Async fixture for test client."""
    async with AsyncAPIClient("http://test.example.com") as client:
        yield client

@pytest.mark.asyncio
async def test_with_async_fixture(async_client):
    """Test using async fixture."""
    # Use the async client fixture
    with patch.object(async_client, 'get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"test": "data"}
        result = await async_client.get("/test")
        assert result["test"] == "data"
```

## Key Takeaways

1. **Use async for I/O-bound operations**: Async programming excels at handling operations that involve waiting (network, file I/O, database operations).

2. **Always await coroutines**: Forgetting `await` is a common mistake that can lead to subtle bugs.

3. **Limit concurrency**: Use semaphores or other mechanisms to prevent overwhelming external services.

4. **Handle errors properly**: Use `return_exceptions=True` in `gather()` when you want to handle individual failures.

5. **Don't block the event loop**: Avoid synchronous blocking operations in async code.

6. **Use async context managers**: They're essential for properly managing resources in async code.

7. **Test thoroughly**: Async code can have subtle timing-related bugs that require careful testing.

Async programming is powerful but requires a different mindset. Start with simple examples and gradually build up to more complex patterns. Remember that async programming is about concurrency, not parallelism - you're doing many things at once, but not necessarily at the same time.
