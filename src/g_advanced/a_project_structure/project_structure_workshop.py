#!/usr/bin/env python3
"""
Project Structure Workshop - Tasks

This workshop provides hands-on exercises for learning Python project organization,
structure patterns, and best practices. Each task builds upon previous concepts
and introduces new organizational challenges.

Complete each task by implementing the required functionality.
Focus on proper structure, clear organization, and maintainable code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

print("🏗️  Python Project Structure Workshop")
print("=" * 50)

# =============================================================================
# TASK 1: Basic Module Organization
# =============================================================================

print("\n📁 TASK 1: Basic Module Organization")
print("-" * 40)

"""
Create a simple calculator package with proper module organization.

Requirements:
1. Create a 'calculator' package with separate modules for:
   - basic_operations.py (add, subtract, multiply, divide)
   - advanced_operations.py (power, square_root, factorial)
   - constants.py (mathematical constants)
   - utils.py (helper functions)

2. Implement the __init__.py file to expose a clean API

3. Each module should have proper docstrings and type hints

TODO: Implement the calculator package structure below
"""

# Task 1.1: Create the basic operations module
def create_basic_operations():
    """
    Create basic arithmetic operations.
    
    Implement: add, subtract, multiply, divide functions
    All functions should have type hints and handle edge cases
    """
    # YOUR CODE HERE
    pass

# Task 1.2: Create advanced operations module  
def create_advanced_operations():
    """
    Create advanced mathematical operations.
    
    Implement: power, square_root, factorial functions
    Include proper error handling for invalid inputs
    """
    # YOUR CODE HERE
    pass

# Task 1.3: Create constants module
def create_constants_module():
    """
    Create a module with mathematical constants.
    
    Include: PI, E, GOLDEN_RATIO, EULER_GAMMA
    Add utility functions to work with these constants
    """
    # YOUR CODE HERE
    pass

# Task 1.4: Create package initialization
def create_package_init():
    """
    Create __init__.py that exposes a clean API.
    
    Should allow: from calculator import add, multiply, PI
    Hide implementation details and internal modules
    """
    # YOUR CODE HERE
    pass

# =============================================================================
# TASK 2: Configuration Management
# =============================================================================

print("\n⚙️  TASK 2: Configuration Management")
print("-" * 40)

"""
Create a flexible configuration system that supports multiple environments
and different configuration sources.

Requirements:
1. Support environment-based configuration (dev, staging, prod)
2. Load configuration from files (JSON, YAML) and environment variables
3. Provide configuration validation
4. Support configuration inheritance and overrides
"""

# Task 2.1: Basic configuration class
@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    # TODO: Add fields for host, port, name, user, password
    # TODO: Add class method to create from environment variables
    pass

@dataclass 
class AppConfig:
    """Main application configuration."""
    # TODO: Add fields for debug, secret_key, database config
    # TODO: Add validation in __post_init__
    # TODO: Add class method for environment-specific configs
    pass

# Task 2.2: Configuration loader
class ConfigurationManager:
    """Manage application configuration from multiple sources."""
    
    def __init__(self, config_dir: str = "config"):
        # TODO: Initialize configuration manager
        pass
    
    def load_from_file(self, filename: str) -> dict[str, Any]:
        """Load configuration from file (JSON or YAML)."""
        # TODO: Implement file loading with format detection
        pass
    
    def load_from_env(self, prefix: str = "APP_") -> dict[str, Any]:
        """Load configuration from environment variables."""
        # TODO: Implement environment variable loading
        pass
    
    def merge_configs(self, *sources: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple configuration sources."""
        # TODO: Implement configuration merging with proper precedence
        pass

# Task 2.3: Configuration validation
def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate configuration and return list of errors."""
    # TODO: Implement configuration validation
    # Check required fields, valid values, etc.
    pass

# =============================================================================
# TASK 3: Package Structure Design
# =============================================================================

print("\n📦 TASK 3: Package Structure Design")
print("-" * 40)

"""
Design and implement a complete package structure for a web application
that handles users, products, and orders.

Requirements:
1. Organize code into logical packages (models, services, utils, api)
2. Implement proper separation of concerns
3. Create clean interfaces between layers
4. Support dependency injection
"""

# Task 3.1: Model layer
class BaseModel(ABC):
    """Base class for all data models."""
    
    def __init__(self, **kwargs): # noqa : B027
        # TODO: Implement base model initialization
        pass
    
    @abstractmethod
    def validate(self) -> list[str]:
        """Validate model data and return errors."""
        pass
    
    def to_dict(self) -> dict[str, Any]: # noqa : B027
        """Convert model to dictionary."""
        # TODO: Implement model serialization
        pass

class User(BaseModel):
    """User model."""
    
    def __init__(self, username: str, email: str, **kwargs):
        # TODO: Implement user initialization with validation
        pass
    
    def validate(self) -> list[str]:
        """Validate user data."""
        # TODO: Implement user validation
        pass

class Product(BaseModel):
    """Product model."""
    
    def __init__(self, name: str, price: float, **kwargs):
        # TODO: Implement product initialization
        pass
    
    def validate(self) -> list[str]:
        """Validate product data."""
        # TODO: Implement product validation
        pass

# Task 3.2: Service layer
class BaseService:
    """Base class for all services."""
    
    def __init__(self, repository=None):
        # TODO: Implement service initialization with dependency injection
        pass

class UserService(BaseService):
    """Service for user-related business logic."""
    
    def create_user(self, user_data: dict[str, Any]) -> User:
        """Create a new user with validation."""
        # TODO: Implement user creation with business logic
        pass
    
    def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user credentials."""
        # TODO: Implement user authentication
        pass

class ProductService(BaseService):
    """Service for product-related business logic."""
    
    def create_product(self, product_data: dict[str, Any]) -> Product:
        """Create a new product."""
        # TODO: Implement product creation
        pass
    
    def calculate_discount(self, product: Product, discount_percent: float) -> float:
        """Calculate discounted price."""
        # TODO: Implement discount calculation
        pass

# Task 3.3: Repository pattern
class Repository(ABC):
    """Abstract repository for data access."""
    
    @abstractmethod
    def save(self, entity: BaseModel) -> BaseModel:
        """Save entity to storage."""
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: int) -> BaseModel | None:
        """Find entity by ID."""
        pass
    
    @abstractmethod
    def find_all(self) -> list[BaseModel]:
        """Find all entities."""
        pass

class InMemoryRepository(Repository):
    """In-memory repository implementation."""
    
    def __init__(self):
        # TODO: Implement in-memory storage
        pass
    
    def save(self, entity: BaseModel) -> BaseModel:
        # TODO: Implement save logic
        pass
    
    def find_by_id(self, entity_id: int) -> BaseModel | None:
        # TODO: Implement find by ID
        pass
    
    def find_all(self) -> list[BaseModel]:
        # TODO: Implement find all
        pass

# =============================================================================
# TASK 4: Plugin Architecture
# =============================================================================

print("\n🔌 TASK 4: Plugin Architecture")
print("-" * 40)

"""
Implement a plugin system that allows extending application functionality
without modifying core code.

Requirements:
1. Create a plugin base class and plugin manager
2. Implement plugin discovery and loading
3. Support plugin configuration and lifecycle management
4. Create example plugins for different functionality
"""

# Task 4.1: Plugin base class
class Plugin(ABC):
    """Base class for all plugins."""
    
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    
    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the plugin's main functionality."""
        pass
    
    def cleanup(self) -> None: # noqa : B027
        """Clean up plugin resources."""
        # TODO: Implement default cleanup
        pass

# Task 4.2: Plugin manager
class PluginManager:
    """Manage and coordinate plugins."""
    
    def __init__(self):
        # TODO: Initialize plugin manager
        pass
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin."""
        # TODO: Implement plugin registration
        pass
    
    def load_plugin(self, name: str, config: dict[str, Any] = None) -> None:
        """Load and initialize a plugin."""
        # TODO: Implement plugin loading
        pass
    
    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        """Execute a plugin by name."""
        # TODO: Implement plugin execution
        pass
    
    def list_plugins(self) -> list[str]:
        """List all registered plugins."""
        # TODO: Return list of plugin names
        pass

# Task 4.3: Example plugins
class EmailPlugin(Plugin):
    """Plugin for sending emails."""
    
    name = "email"
    version = "1.0.0"
    description = "Send emails via SMTP"
    
    def initialize(self, config: dict[str, Any]) -> None:
        # TODO: Initialize email configuration
        pass
    
    def execute(self, to: str, subject: str, body: str) -> bool:
        # TODO: Implement email sending
        pass

class LoggingPlugin(Plugin):
    """Plugin for advanced logging."""
    
    name = "logging"
    version = "1.0.0"
    description = "Enhanced logging functionality"
    
    def initialize(self, config: dict[str, Any]) -> None:
        # TODO: Initialize logging configuration
        pass
    
    def execute(self, level: str, message: str, **kwargs) -> None:
        # TODO: Implement enhanced logging
        pass

# =============================================================================
# TASK 5: Dependency Injection Container
# =============================================================================

print("\n💉 TASK 5: Dependency Injection Container")
print("-" * 40)

"""
Create a dependency injection container to manage object creation
and dependencies between components.

Requirements:
1. Support singleton and transient lifetimes
2. Allow registration of services and factories
3. Resolve dependencies automatically
4. Support circular dependency detection
"""

# Task 5.1: Service container
class ServiceContainer:
    """Container for dependency injection."""
    
    def __init__(self):
        # TODO: Initialize container with registrations storage
        pass
    
    def register_singleton(self, interface: type, implementation: type) -> None:
        """Register a singleton service."""
        # TODO: Implement singleton registration
        pass
    
    def register_transient(self, interface: type, implementation: type) -> None:
        """Register a transient service."""
        # TODO: Implement transient registration
        pass
    
    def register_instance(self, interface: type, instance: Any) -> None:
        """Register a specific instance."""
        # TODO: Implement instance registration
        pass
    
    def resolve(self, service_type: type) -> Any:
        """Resolve a service and its dependencies."""
        # TODO: Implement service resolution with dependency injection
        pass

# Task 5.2: Service decorators
def singleton(interface: type):
    """Decorator to register a class as singleton."""
    def decorator(cls):
        # TODO: Implement singleton decorator
        return cls
    return decorator

def transient(interface: type):
    """Decorator to register a class as transient."""
    def decorator(cls):
        # TODO: Implement transient decorator
        return cls
    return decorator

# Task 5.3: Example services with dependencies
class DatabaseConnection:
    """Example database connection service."""
    
    def __init__(self, connection_string: str):
        # TODO: Initialize database connection
        pass

class UserRepository:
    """User repository with database dependency."""
    
    def __init__(self, db_connection: DatabaseConnection):
        # TODO: Initialize repository with database dependency
        pass

class UserService: # noqa : F811
    """User service with repository dependency."""
    
    def __init__(self, user_repository: UserRepository):
        # TODO: Initialize service with repository dependency
        pass

# =============================================================================
# TASK 6: Testing Structure
# =============================================================================

print("\n🧪 TASK 6: Testing Structure")
print("-" * 40)

"""
Create a comprehensive testing structure that supports unit tests,
integration tests, and test fixtures.

Requirements:
1. Organize tests to mirror source structure
2. Create reusable test fixtures
3. Implement test utilities and helpers
4. Support different test categories
"""

# Task 6.1: Test fixtures
class TestFixtures:
    """Reusable test fixtures."""
    
    @staticmethod
    def create_test_user(**kwargs) -> User:
        """Create a test user with default values."""
        # TODO: Implement test user creation
        pass
    
    @staticmethod
    def create_test_product(**kwargs) -> Product:
        """Create a test product with default values."""
        # TODO: Implement test product creation
        pass
    
    @staticmethod
    def create_test_config(**kwargs) -> AppConfig:
        """Create test configuration."""
        # TODO: Implement test configuration
        pass

# Task 6.2: Test base classes
class BaseTestCase:
    """Base class for all test cases."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # TODO: Implement test setup
        pass
    
    def teardown_method(self):
        """Clean up after each test."""
        # TODO: Implement test cleanup
        pass

class IntegrationTestCase(BaseTestCase):
    """Base class for integration tests."""
    
    def setup_method(self):
        """Set up integration test environment."""
        # TODO: Implement integration test setup
        super().setup_method()

# Task 6.3: Test utilities
class TestDatabase:
    """Test database utilities."""
    
    @staticmethod
    def create_test_database():
        """Create a test database."""
        # TODO: Implement test database creation
        pass
    
    @staticmethod
    def clear_database():
        """Clear test database."""
        # TODO: Implement database clearing
        pass

class TestHelpers:
    """General test helper functions."""
    
    @staticmethod
    def assert_valid_email(email: str) -> bool:
        """Assert that email format is valid."""
        # TODO: Implement email validation assertion
        pass
    
    @staticmethod
    def generate_test_data(count: int, data_type: str) -> list[Any]:
        """Generate test data of specified type."""
        # TODO: Implement test data generation
        pass

# =============================================================================
# WORKSHOP COMPLETION
# =============================================================================

def run_workshop_tests():
    """Run tests to verify workshop completion."""
    print("\n✅ Testing Workshop Implementation")
    print("-" * 40)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Module organization
    try:
        # TODO: Add tests for Task 1
        print("✅ Task 1: Module Organization - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Task 1: Module Organization - FAILED: {e}")
    
    # Test 2: Configuration management
    try:
        # TODO: Add tests for Task 2
        print("✅ Task 2: Configuration Management - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Task 2: Configuration Management - FAILED: {e}")
    
    # Test 3: Package structure
    try:
        # TODO: Add tests for Task 3
        print("✅ Task 3: Package Structure - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Task 3: Package Structure - FAILED: {e}")
    
    # Test 4: Plugin architecture
    try:
        # TODO: Add tests for Task 4
        print("✅ Task 4: Plugin Architecture - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Task 4: Plugin Architecture - FAILED: {e}")
    
    # Test 5: Dependency injection
    try:
        # TODO: Add tests for Task 5
        print("✅ Task 5: Dependency Injection - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Task 5: Dependency Injection - FAILED: {e}")
    
    # Test 6: Testing structure
    try:
        # TODO: Add tests for Task 6
        print("✅ Task 6: Testing Structure - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Task 6: Testing Structure - FAILED: {e}")
    
    print(f"\n📊 Workshop Results: {tests_passed}/{total_tests} tasks completed")
    
    if tests_passed == total_tests:
        print("🎉 Congratulations! You've completed the Project Structure Workshop!")
        print("You now understand how to organize Python projects for maintainability and scale.")
    else:
        print("📚 Review the failed tasks and try again. Good project structure is essential!")

if __name__ == "__main__":
    print("\n🎯 Workshop Instructions:")
    print("1. Complete each task by implementing the required functionality")
    print("2. Focus on proper organization, clear interfaces, and maintainable code")
    print("3. Run the tests to verify your implementation")
    print("4. Refer to the project_structure.md guide for detailed explanations")
    
    # Uncomment to run tests
    # run_workshop_tests()
