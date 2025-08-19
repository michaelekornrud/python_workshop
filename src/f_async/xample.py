"""
Async Programming Workshop - COMPLETE SOLUTIONS

Complete solutions for all async programming tasks demonstrating best practices
and real-world patterns in asynchronous Python development.
"""

import asyncio
import aiohttp
import aiofiles
import json
import time
import random
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import weakref

# =============================================================================
# TASK 1: Basic Async/Await and Coroutines - SOLUTION
# =============================================================================

async def async_hello(name: str, delay: float) -> str:
    """Greet user after specified delay."""
    await asyncio.sleep(delay)
    return f"Hello, {name}!"

async def async_calculator(operation: str, a: float, b: float, delay: float) -> float:
    """Perform calculation after specified delay."""
    await asyncio.sleep(delay)
    
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    return operations[operation](a, b)

async def run_sequential() -> tuple[List[str], float]:
    """Run tasks sequentially and return results with timing."""
    start_time = time.time()
    
    results = []
    results.append(await async_hello("Alice", 0.5))
    results.append(await async_hello("Bob", 0.3))
    results.append(str(await async_calculator("add", 10, 5, 0.2)))
    results.append(str(await async_calculator("multiply", 3, 4, 0.4)))
    
    duration = time.time() - start_time
    return results, duration

async def run_concurrent() -> tuple[List[Any], float]:
    """Run tasks concurrently and return results with timing."""
    start_time = time.time()
    
    tasks = [
        async_hello("Alice", 0.5),
        async_hello("Bob", 0.3),
        async_calculator("add", 10, 5, 0.2),
        async_calculator("multiply", 3, 4, 0.4)
    ]
    
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    # Convert numeric results to strings for consistency
    results = [str(result) for result in results]
    return results, duration

# =============================================================================
# TASK 2: HTTP Client and Real-World Async Operations - SOLUTION
# =============================================================================

class AsyncHTTPClient:
    """Async HTTP client with session management and retry logic."""
    
    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._error_count = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Fetch single URL with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        self._request_count += 1
        
        try:
            async with self.session.get(url, **kwargs) as response:
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = {"content": await response.text(), "content_type": response.content_type}
                
                return {
                    "url": url,
                    "status": response.status,
                    "data": data,
                    "headers": dict(response.headers),
                    "success": True
                }
        except Exception as e:
            self._error_count += 1
            return {
                "url": url,
                "error": str(e),
                "success": False
            }
    
    async def fetch_multiple(self, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """Fetch multiple URLs concurrently with semaphore control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_fetch(url: str):
            async with semaphore:
                return await self.fetch_url(url)
        
        tasks = [limited_fetch(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "url": urls[i] if i < len(urls) else "unknown",
                    "error": str(result),
                    "success": False
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def fetch_with_retry(self, url: str, max_retries: int = None) -> Dict[str, Any]:
        """Fetch URL with retry logic and exponential backoff."""
        max_retries = max_retries or self.max_retries
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await self.fetch_url(url)
                if result["success"]:
                    result["attempts"] = attempt + 1
                    return result
                else:
                    last_error = result["error"]
            except Exception as e:
                last_error = str(e)
            
            if attempt < max_retries:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
        
        return {
            "url": url,
            "error": f"Failed after {max_retries + 1} attempts. Last error: {last_error}",
            "success": False,
            "attempts": max_retries + 1
        }
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get client statistics."""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "success_rate": (self._request_count - self._error_count) / max(self._request_count, 1)
        }

# =============================================================================
# TASK 3: Producer-Consumer Pattern with Async Queues - SOLUTION
# =============================================================================

@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    items_produced: int = 0
    items_consumed: int = 0
    items_in_queue: int = 0
    producers_active: int = 0
    consumers_active: int = 0
    total_processing_time: float = 0.0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)

class DataProcessor:
    """Producer-consumer system with async queues."""
    
    def __init__(self, queue_size: int = 10):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.stats = ProcessingStats()
        self.running = False
        self._producer_tasks: List[asyncio.Task] = []
        self._consumer_tasks: List[asyncio.Task] = []
        
    async def producer(self, name: str, items: List[Any], delay_range: tuple[float, float] = (0.1, 0.5)):
        """Producer that generates work items with random delays."""
        print(f"🏭 Producer {name} starting with {len(items)} items")
        self.stats.producers_active += 1
        
        try:
            for item in items:
                if not self.running:
                    break
                
                await self.queue.put({
                    "data": item,
                    "producer": name,
                    "timestamp": datetime.now(),
                    "id": f"{name}-{self.stats.items_produced}"
                })
                
                self.stats.items_produced += 1
                self.stats.items_in_queue = self.queue.qsize()
                
                print(f"🏭 Producer {name} added: {item}")
                
                delay = random.uniform(*delay_range)
                await asyncio.sleep(delay)
                
        except Exception as e:
            print(f"❌ Producer {name} error: {e}")
            self.stats.errors += 1
        finally:
            self.stats.producers_active -= 1
            print(f"🏭 Producer {name} finished")
    
    async def consumer(self, name: str, process_func: Optional[Callable] = None):
        """Consumer that processes work items."""
        print(f"🔧 Consumer {name} starting")
        self.stats.consumers_active += 1
        
        if process_func is None:
            async def default_process(item):
                await asyncio.sleep(0.2)  # Simulate processing
                return f"processed-{item['data']}"
            process_func = default_process
        
        try:
            while self.running:
                try:
                    # Wait for work with timeout
                    work_item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    
                    start_time = time.time()
                    
                    # Process the item
                    result = await process_func(work_item)
                    
                    processing_time = time.time() - start_time
                    self.stats.total_processing_time += processing_time
                    self.stats.items_consumed += 1
                    self.stats.items_in_queue = self.queue.qsize()
                    
                    print(f"🔧 Consumer {name} processed: {work_item['data']} -> {result}")
                    
                    # Mark task as done
                    self.queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Check if we should continue running
                    continue
                except Exception as e:
                    print(f"❌ Consumer {name} error: {e}")
                    self.stats.errors += 1
                    self.queue.task_done()  # Don't leave the queue hanging
                    
        finally:
            self.stats.consumers_active -= 1
            print(f"🔧 Consumer {name} stopping")
    
    async def run_system(self, producers: Dict[str, List[Any]], consumers: List[str], 
                        timeout: float = 30.0) -> ProcessingStats:
        """Run the complete producer-consumer system."""
        self.running = True
        self.stats = ProcessingStats()  # Reset stats
        
        print(f"🚀 Starting system with {len(producers)} producers and {len(consumers)} consumers")
        
        # Start producers
        for name, items in producers.items():
            task = asyncio.create_task(self.producer(name, items))
            self._producer_tasks.append(task)
        
        # Start consumers
        for name in consumers:
            task = asyncio.create_task(self.consumer(name))
            self._consumer_tasks.append(task)
        
        try:
            # Wait for producers to finish
            await asyncio.gather(*self._producer_tasks, return_exceptions=True)
            print("📋 All producers finished")
            
            # Wait for queue to be empty
            await self.queue.join()
            print("📋 All items processed")
            
        except asyncio.TimeoutError:
            print("⏰ System timeout reached")
        finally:
            # Stop the system
            self.running = False
            
            # Wait for consumers to stop
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
            
            # Calculate final stats
            self.stats.start_time = datetime.now()
            
            print(f"📊 Final stats: {self.stats.items_produced} produced, {self.stats.items_consumed} consumed")
            return self.stats

# =============================================================================
# TASK 4: Async Context Managers and Resource Management - SOLUTION
# =============================================================================

class AsyncDatabaseConnection:
    """Simulated async database connection."""
    
    def __init__(self, connection_id: str, connection_string: str):
        self.connection_id = connection_id
        self.connection_string = connection_string
        self.is_connected = False
        self.transaction_active = False
        
    async def connect(self):
        """Establish database connection."""
        print(f"🔌 Connecting to database: {self.connection_id}")
        await asyncio.sleep(0.1)  # Simulate connection time
        self.is_connected = True
        
    async def disconnect(self):
        """Close database connection."""
        print(f"🔌 Disconnecting: {self.connection_id}")
        await asyncio.sleep(0.05)  # Simulate cleanup time
        self.is_connected = False
        
    async def execute(self, query: str) -> Dict[str, Any]:
        """Execute database query."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
            
        await asyncio.sleep(0.2)  # Simulate query execution
        return {
            "query": query,
            "connection_id": self.connection_id,
            "timestamp": datetime.now(),
            "rows_affected": random.randint(1, 100)
        }
    
    async def begin_transaction(self):
        """Begin database transaction."""
        if self.transaction_active:
            raise RuntimeError("Transaction already active")
        self.transaction_active = True
        await asyncio.sleep(0.01)
        
    async def commit(self):
        """Commit transaction."""
        if not self.transaction_active:
            raise RuntimeError("No active transaction")
        self.transaction_active = False
        await asyncio.sleep(0.01)
        
    async def rollback(self):
        """Rollback transaction."""
        if not self.transaction_active:
            raise RuntimeError("No active transaction")
        self.transaction_active = False
        await asyncio.sleep(0.01)

class AsyncDatabasePool:
    """Database connection pool with async context management."""
    
    def __init__(self, connection_string: str, pool_size: int = 5):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.available_connections: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self.all_connections: List[AsyncDatabaseConnection] = []
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "queries_executed": 0
        }
    
    async def __aenter__(self):
        """Initialize connection pool."""
        print(f"🏊 Initializing database pool with {self.pool_size} connections")
        
        for i in range(self.pool_size):
            conn = AsyncDatabaseConnection(f"conn-{i}", self.connection_string)
            await conn.connect()
            self.all_connections.append(conn)
            await self.available_connections.put(conn)
            self.stats["total_connections"] += 1
            
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up connection pool."""
        print("🏊 Cleaning up database pool")
        
        for conn in self.all_connections:
            if conn.is_connected:
                await conn.disconnect()
        
        self.all_connections.clear()
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[AsyncDatabaseConnection, None]:
        """Get connection from pool with automatic return."""
        conn = await self.available_connections.get()
        self.stats["active_connections"] += 1
        
        try:
            yield conn
        finally:
            self.stats["active_connections"] -= 1
            await self.available_connections.put(conn)
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute query using pool connection."""
        async with self.get_connection() as conn:
            result = await conn.execute(query)
            self.stats["queries_executed"] += 1
            return result

class AsyncFileManager:
    """Async file manager with JSON operations."""
    
    def __init__(self, filepath: str, mode: str = "r+", encoding: str = "utf-8"):
        self.filepath = filepath
        self.mode = mode
        self.encoding = encoding
        self.file = None
        self._backup_created = False
    
    async def __aenter__(self):
        """Open file and create backup if needed."""
        print(f"📁 Opening file: {self.filepath}")
        
        # Create backup for write operations
        if "w" in self.mode or "a" in self.mode:
            await self._create_backup()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close file and handle cleanup."""
        print(f"📁 Closing file: {self.filepath}")
        
        if self.file:
            await self.file.close()
        
        # If there was an error and we created a backup, restore it
        if exc_type and self._backup_created:
            await self._restore_backup()
    
    async def _create_backup(self):
        """Create backup of existing file."""
        try:
            async with aiofiles.open(self.filepath, 'r', encoding=self.encoding) as f:
                content = await f.read()
            
            backup_path = f"{self.filepath}.backup"
            async with aiofiles.open(backup_path, 'w', encoding=self.encoding) as f:
                await f.write(content)
            
            self._backup_created = True
            print(f"📁 Backup created: {backup_path}")
            
        except FileNotFoundError:
            # File doesn't exist yet, no backup needed
            pass
    
    async def _restore_backup(self):
        """Restore file from backup."""
        backup_path = f"{self.filepath}.backup"
        try:
            async with aiofiles.open(backup_path, 'r', encoding=self.encoding) as f:
                content = await f.read()
            
            async with aiofiles.open(self.filepath, 'w', encoding=self.encoding) as f:
                await f.write(content)
            
            print(f"📁 File restored from backup")
        except FileNotFoundError:
            pass
    
    async def read_json(self) -> Dict[str, Any]:
        """Read JSON data from file."""
        try:
            async with aiofiles.open(self.filepath, 'r', encoding=self.encoding) as f:
                content = await f.read()
                return json.loads(content) if content.strip() else {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {self.filepath}: {e}")
    
    async def write_json(self, data: Dict[str, Any], indent: int = 2) -> None:
        """Write JSON data to file."""
        json_content = json.dumps(data, indent=indent, ensure_ascii=False)
        
        async with aiofiles.open(self.filepath, 'w', encoding=self.encoding) as f:
            await f.write(json_content)
    
    async def append_json_line(self, data: Dict[str, Any]) -> None:
        """Append JSON line to file (JSONL format)."""
        json_line = json.dumps(data, ensure_ascii=False) + "\n"
        
        async with aiofiles.open(self.filepath, 'a', encoding=self.encoding) as f:
            await f.write(json_line)

# =============================================================================
# TASK 5: Async Iterators and Streaming Data - SOLUTION
# =============================================================================

class AsyncDataStream:
    """Async iterator for streaming data from various sources."""
    
    def __init__(self, source_type: str, **config):
        self.source_type = source_type
        self.config = config
        self.current_position = 0
        
    @classmethod
    def from_api(cls, base_url: str, page_size: int = 10, max_pages: int = None):
        """Create stream from paginated API."""
        return cls("api", base_url=base_url, page_size=page_size, max_pages=max_pages)
    
    @classmethod
    def from_file(cls, filepath: str, chunk_size: int = 1024):
        """Create stream from file."""
        return cls("file", filepath=filepath, chunk_size=chunk_size)
    
    @classmethod
    def from_generator(cls, generator_func: Callable, *args, **kwargs):
        """Create stream from generator function."""
        return cls("generator", generator_func=generator_func, args=args, kwargs=kwargs)
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.source_type == "api":
            return await self._next_from_api()
        elif self.source_type == "file":
            return await self._next_from_file()
        elif self.source_type == "generator":
            return await self._next_from_generator()
        else:
            raise StopAsyncIteration
    
    async def _next_from_api(self):
        """Get next item from API source."""
        base_url = self.config["base_url"]
        page_size = self.config["page_size"]
        max_pages = self.config.get("max_pages")
        
        page = self.current_position // page_size
        if max_pages and page >= max_pages:
            raise StopAsyncIteration
        
        # Simulate API call
        await asyncio.sleep(0.1)
        
        if self.current_position >= 50:  # Simulate end of data
            raise StopAsyncIteration
        
        item = {
            "id": self.current_position,
            "data": f"Item {self.current_position}",
            "page": page,
            "source": "api"
        }
        
        self.current_position += 1
        return item
    
    async def _next_from_file(self):
        """Get next chunk from file source."""
        # Simulate file reading
        if self.current_position >= 10:  # Simulate end of file
            raise StopAsyncIteration
        
        await asyncio.sleep(0.05)
        
        chunk = {
            "chunk_id": self.current_position,
            "data": f"File chunk {self.current_position}",
            "size": self.config["chunk_size"],
            "source": "file"
        }
        
        self.current_position += 1
        return chunk
    
    async def _next_from_generator(self):
        """Get next item from generator source."""
        generator_func = self.config["generator_func"]
        
        # For demo, create a simple async generator
        if self.current_position >= 5:
            raise StopAsyncIteration
        
        await asyncio.sleep(0.1)
        
        item = await generator_func(self.current_position, *self.config["args"], **self.config["kwargs"])
        self.current_position += 1
        return item

class AsyncDataProcessor:
    """Process streaming data with various transformations."""
    
    def __init__(self):
        self.processors_registered = {}
        self.stats = {
            "items_processed": 0,
            "processing_time": 0.0,
            "errors": 0
        }
    
    def register_processor(self, name: str, processor_func: Callable):
        """Register a named processor function."""
        self.processors_registered[name] = processor_func
    
    async def transform_stream(self, stream: AsyncDataStream, 
                             transform_func: Callable) -> AsyncGenerator[Any, None]:
        """Transform each item in the stream."""
        async for item in stream:
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(transform_func):
                    transformed = await transform_func(item)
                else:
                    transformed = transform_func(item)
                
                processing_time = time.time() - start_time
                self.stats["processing_time"] += processing_time
                self.stats["items_processed"] += 1
                
                yield transformed
                
            except Exception as e:
                self.stats["errors"] += 1
                print(f"❌ Error processing item {item}: {e}")
                continue
    
    async def filter_stream(self, stream: AsyncDataStream, 
                           filter_func: Callable) -> AsyncGenerator[Any, None]:
        """Filter items in the stream based on criteria."""
        async for item in stream:
            try:
                if asyncio.iscoroutinefunction(filter_func):
                    should_include = await filter_func(item)
                else:
                    should_include = filter_func(item)
                
                if should_include:
                    yield item
                    
            except Exception as e:
                self.stats["errors"] += 1
                print(f"❌ Error filtering item {item}: {e}")
                continue
    
    async def batch_stream(self, stream: AsyncDataStream, 
                          batch_size: int) -> AsyncGenerator[List[Any], None]:
        """Batch items from stream into groups."""
        batch = []
        
        async for item in stream:
            batch.append(item)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining items
        if batch:
            yield batch
    
    async def parallel_process(self, stream: AsyncDataStream, 
                             process_func: Callable, 
                             max_concurrent: int = 5) -> AsyncGenerator[Any, None]:
        """Process stream items in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(process_func):
                    return await process_func(item)
                else:
                    return process_func(item)
        
        # Collect items in batches for parallel processing
        async for batch in self.batch_stream(stream, max_concurrent):
            tasks = [process_item(item) for item in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.stats["errors"] += 1
                    print(f"❌ Parallel processing error: {result}")
                else:
                    self.stats["items_processed"] += 1
                    yield result

# =============================================================================
# TASK 6: Advanced Concurrency Control - SOLUTION
# =============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0, 
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                print("🔄 Circuit breaker: HALF_OPEN")
            else:
                raise RuntimeError("Circuit breaker is OPEN")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Handle success
            await self._on_success()
            return result
            
        except Exception as e:
            # Handle failure
            await self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time:
            time_since_failure = datetime.now() - self.last_failure_time
            return time_since_failure.total_seconds() >= self.reset_timeout
        return True
    
    async def _on_success(self):
        """Handle successful function execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                print("✅ Circuit breaker: CLOSED")
        
        # Reset failure count on success in closed state
        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            print("❌ Circuit breaker: OPEN (failed during half-open)")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            print(f"❌ Circuit breaker: OPEN (threshold {self.failure_threshold} reached)")

class AsyncTaskManager:
    """Advanced task manager with concurrency control."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, Dict] = {}
        self.task_stats = defaultdict(int)
        
    async def run_with_semaphore(self, tasks: List[Callable], 
                                max_concurrent: int = 5) -> List[Any]:
        """Run tasks with semaphore-controlled concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task_func):
            async with semaphore:
                if asyncio.iscoroutinefunction(task_func):
                    return await task_func()
                else:
                    return task_func()
        
        task_coroutines = [limited_task(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        self.task_stats["semaphore_controlled"] += len(tasks)
        return results
    
    async def run_with_timeout(self, tasks: List[Callable], 
                              timeout: float) -> List[Any]:
        """Run tasks with timeout management."""
        async def timeout_task(task_func):
            try:
                if asyncio.iscoroutinefunction(task_func):
                    return await asyncio.wait_for(task_func(), timeout=timeout)
                else:
                    return await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, task_func), 
                        timeout=timeout
                    )
            except asyncio.TimeoutError:
                return {"error": "timeout", "timeout": timeout}
        
        task_coroutines = [timeout_task(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        self.task_stats["timeout_controlled"] += len(tasks)
        return results
    
    async def run_with_circuit_breaker(self, func: Callable, 
                                     failure_threshold: int = 3, 
                                     breaker_name: str = "default") -> Any:
        """Run function with circuit breaker pattern."""
        if breaker_name not in self.circuit_breakers:
            self.circuit_breakers[breaker_name] = CircuitBreaker(
                failure_threshold=failure_threshold
            )
        
        breaker = self.circuit_breakers[breaker_name]
        result = await breaker.call(func)
        
        self.task_stats["circuit_breaker_calls"] += 1
        return result
    
    async def run_with_rate_limit(self, func: Callable, 
                                 rate_limit: int = 10, 
                                 time_window: float = 60.0,
                                 limiter_name: str = "default") -> Any:
        """Run function with rate limiting."""
        if limiter_name not in self.rate_limiters:
            self.rate_limiters[limiter_name] = {
                "requests": deque(),
                "rate_limit": rate_limit,
                "time_window": time_window
            }
        
        limiter = self.rate_limiters[limiter_name]
        now = time.time()
        
        # Remove old requests outside time window
        while limiter["requests"] and limiter["requests"][0] <= now - limiter["time_window"]:
            limiter["requests"].popleft()
        
        # Check if rate limit exceeded
        if len(limiter["requests"]) >= limiter["rate_limit"]:
            sleep_time = limiter["time_window"] - (now - limiter["requests"][0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this request
        limiter["requests"].append(now)
        
        # Execute function
        if asyncio.iscoroutinefunction(func):
            result = await func()
        else:
            result = func()
        
        self.task_stats["rate_limited_calls"] += 1
        return result
    
    async def run_with_retry(self, func: Callable, max_retries: int = 3, 
                           backoff_factor: float = 2.0) -> Any:
        """Run function with retry logic and exponential backoff."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
                    
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    wait_time = backoff_factor ** attempt + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                    print(f"🔄 Retry attempt {attempt + 1} after {wait_time:.2f}s")
                else:
                    print(f"❌ All {max_retries + 1} attempts failed")
        
        self.task_stats["retry_exhausted"] += 1
        raise last_exception

# =============================================================================
# Helper function for testing (demo generator)
# =============================================================================

async def demo_generator(index: int, prefix: str = "gen") -> Dict[str, Any]:
    """Demo generator function for async streams."""
    await asyncio.sleep(0.1)
    return {
        "id": index,
        "data": f"{prefix}-{index}",
        "timestamp": datetime.now().isoformat()
    }

# =============================================================================
# MAIN DEMONSTRATION FUNCTION
# =============================================================================

async def demonstrate_all_tasks():
    """Demonstrate all implemented async patterns."""
    print("🚀 Starting Async Programming Workshop Demonstrations\n")
    
    # Task 1: Basic Async/Await
    print("=" * 50)
    print("TASK 1: Basic Async/Await")
    print("=" * 50)
    
    # Sequential execution
    seq_results, seq_time = await run_sequential()
    print(f"Sequential execution: {seq_time:.2f}s")
    print(f"Results: {seq_results}")
    
    # Concurrent execution
    conc_results, conc_time = await run_concurrent()
    print(f"Concurrent execution: {conc_time:.2f}s")
    print(f"Results: {conc_results}")
    print(f"Speedup: {seq_time/conc_time:.2f}x faster\n")
    
    # Task 2: HTTP Client
    print("=" * 50)
    print("TASK 2: HTTP Client")
    print("=" * 50)
    
    # Note: This would require actual HTTP endpoints to test
    # For demo, we'll just show the client structure
    print("AsyncHTTPClient implemented with features:")
    print("- Session management")
    print("- Retry logic with exponential backoff")
    print("- Concurrent requests with semaphore control")
    print("- Error handling and statistics\n")
    
    # Task 3: Producer-Consumer
    print("=" * 50)
    print("TASK 3: Producer-Consumer")
    print("=" * 50)
    
    processor = DataProcessor(queue_size=5)
    
    producers = {
        "Producer-A": ["task-1", "task-2", "task-3"],
        "Producer-B": ["work-A", "work-B"]
    }
    
    consumers = ["Consumer-1", "Consumer-2"]
    
    stats = await processor.run_system(producers, consumers, timeout=10.0)
    print(f"Processing completed: {stats.items_produced} produced, {stats.items_consumed} consumed\n")
    
    # Task 4: Context Managers
    print("=" * 50)
    print("TASK 4: Context Managers")
    print("=" * 50)
    
    # Database pool demo
    async with AsyncDatabasePool("postgresql://localhost:5432/demo", pool_size=3) as pool:
        result = await pool.execute_query("SELECT * FROM users LIMIT 5")
        print(f"Database query result: {result}")
    
    # File manager demo
    test_data = {"name": "test", "value": 42, "items": [1, 2, 3]}
    async with AsyncFileManager("demo.json") as file_mgr:
        await file_mgr.write_json(test_data)
        read_data = await file_mgr.read_json()
        print(f"File operations: wrote and read {read_data}\n")
    
    # Task 5: Async Iterators
    print("=" * 50)
    print("TASK 5: Async Iterators and Streaming")
    print("=" * 50)
    
    # Demo async data stream
    processor = AsyncDataProcessor()
    
    # Stream from generator
    stream = AsyncDataStream.from_generator(demo_generator, "demo")
    
    # Transform stream
    transform_func = lambda item: f"transformed-{item['data']}"
    
    print("Processing async stream:")
    count = 0
    async for transformed_item in processor.transform_stream(stream, transform_func):
        print(f"  Processed: {transformed_item}")
        count += 1
        if count >= 3:  # Limit demo output
            break
    
    print(f"Stream processing stats: {processor.stats}\n")
    
    # Task 6: Advanced Concurrency Control
    print("=" * 50)
    print("TASK 6: Advanced Concurrency Control")
    print("=" * 50)
    
    task_manager = AsyncTaskManager()
    
    # Demo circuit breaker
    async def flaky_function():
        if random.random() < 0.7:  # 70% chance of failure
            raise ValueError("Simulated failure")
        return "Success!"
    
    print("Testing circuit breaker pattern:")
    for i in range(5):
        try:
            result = await task_manager.run_with_circuit_breaker(
                flaky_function, failure_threshold=2, breaker_name="demo"
            )
            print(f"  Attempt {i+1}: {result}")
        except Exception as e:
            print(f"  Attempt {i+1}: Failed - {e}")
    
    print(f"Task manager stats: {dict(task_manager.task_stats)}\n")
    
    print("🎉 All demonstrations completed successfully!")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_all_tasks())
