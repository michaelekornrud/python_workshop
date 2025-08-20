#!/usr/bin/env python3
"""
Project Structure Workshop - COMPLETE SOLUTIONS

Complete solutions for all project structure tasks demonstrating best practices
for Python project organization, architecture patterns, and maintainable code structure.
"""

import os
import json
import yaml
import logging
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import weakref
import threading
from collections import defaultdict

# =============================================================================
# TASK 1: Basic Module Organization - SOLUTION
# =============================================================================

# Task 1.1: Basic Operations Module
class BasicOperations:
    """Basic arithmetic operations with proper error handling."""
    
    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        return a + b
    
    @staticmethod
    def subtract(a: float, b: float) -> float:
        """Subtract b from a.
        
        Args:
            a: Number to subtract from
            b: Number to subtract
            
        Returns:
            Difference of a and b
        """
        return a - b
    
    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        return a * b
    
    @staticmethod
    def divide(a: float, b: float) -> float:
        """Divide a by b.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            Quotient of a and b
            
        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# Task 1.2: Advanced Operations Module
import math

class AdvancedOperations:
    """Advanced mathematical operations with error handling."""
    
    @staticmethod
    def power(base: float, exponent: float) -> float:
        """Raise base to the power of exponent.
        
        Args:
            base: Base number
            exponent: Exponent
            
        Returns:
            base raised to the power of exponent
        """
        return base ** exponent
    
    @staticmethod
    def square_root(value: float) -> float:
        """Calculate square root of a number.
        
        Args:
            value: Number to find square root of
            
        Returns:
            Square root of value
            
        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(value)
    
    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial of a number.
        
        Args:
            n: Non-negative integer
            
        Returns:
            Factorial of n
            
        Raises:
            ValueError: If n is negative or not an integer
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("Factorial requires a non-negative integer")
        return math.factorial(n)
    
    @staticmethod
    def logarithm(value: float, base: float = math.e) -> float:
        """Calculate logarithm of value with given base.
        
        Args:
            value: Number to find logarithm of
            base: Base of logarithm (default: e)
            
        Returns:
            Logarithm of value with given base
            
        Raises:
            ValueError: If value <= 0 or base <= 0 or base == 1
        """
        if value <= 0:
            raise ValueError("Logarithm requires positive value")
        if base <= 0 or base == 1:
            raise ValueError("Logarithm base must be positive and not equal to 1")
        
        if base == math.e:
            return math.log(value)
        else:
            return math.log(value) / math.log(base)

# Task 1.3: Constants Module
class MathConstants:
    """Mathematical constants and utility functions."""
    
    PI = math.pi
    E = math.e
    GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
    EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant
    
    @classmethod
    def get_constant(cls, name: str) -> float:
        """Get a mathematical constant by name.
        
        Args:
            name: Name of the constant
            
        Returns:
            Value of the constant
            
        Raises:
            ValueError: If constant name is not found
        """
        constants = {
            'pi': cls.PI,
            'e': cls.E,
            'golden_ratio': cls.GOLDEN_RATIO,
            'euler_gamma': cls.EULER_GAMMA
        }
        
        if name.lower() not in constants:
            raise ValueError(f"Unknown constant: {name}")
        
        return constants[name.lower()]
    
    @classmethod
    def list_constants(cls) -> list[str]:
        """List all available constants."""
        return ['pi', 'e', 'golden_ratio', 'euler_gamma']

# Task 1.4: Calculator Package Implementation
class Calculator:
    """Main calculator interface that combines all operations."""
    
    def __init__(self):
        self.basic = BasicOperations()
        self.advanced = AdvancedOperations()
        self.constants = MathConstants()
        self.history: list[str] = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = self.basic.add(a, b)
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        result = self.basic.subtract(a, b)
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = self.basic.multiply(a, b)
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        result = self.basic.divide(a, b)
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        result = self.advanced.power(base, exponent)
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result
    
    def sqrt(self, value: float) -> float:
        """Calculate square root."""
        result = self.advanced.square_root(value)
        self.history.append(f"sqrt({value}) = {result}")
        return result
    
    def get_history(self) -> list[str]:
        """Get calculation history."""
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear calculation history."""
        self.history.clear()

# =============================================================================
# TASK 2: Configuration Management - SOLUTION
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    name: str = "myapp"
    user: str = "postgres"
    password: str = ""
    max_connections: int = 10
    timeout: float = 30.0
    
    @classmethod
    def from_env(cls, prefix: str = "DB_") -> "DatabaseConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv(f"{prefix}HOST", "localhost"),
            port=int(os.getenv(f"{prefix}PORT", "5432")),
            name=os.getenv(f"{prefix}NAME", "myapp"),
            user=os.getenv(f"{prefix}USER", "postgres"),
            password=os.getenv(f"{prefix}PASSWORD", ""),
            max_connections=int(os.getenv(f"{prefix}MAX_CONNECTIONS", "10")),
            timeout=float(os.getenv(f"{prefix}TIMEOUT", "30.0"))
        )
    
    def to_connection_string(self) -> str:
        """Generate database connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

@dataclass
class CacheConfig:
    """Cache configuration settings."""
    enabled: bool = True
    backend: str = "memory"
    ttl: int = 3600
    max_size: int = 1000
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheConfig":
        """Create configuration from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            backend=data.get("backend", "memory"),
            ttl=data.get("ttl", 3600),
            max_size=data.get("max_size", 1000)
        )

@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    secret_key: str = "dev-key"
    environment: str = "development"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.secret_key or self.secret_key == "dev-key" and self.environment == "production":
            errors.append("Secret key must be set for production environment")
        
        if self.environment not in ["development", "testing", "staging", "production"]:
            errors.append(f"Invalid environment: {self.environment}")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append(f"Invalid log level: {self.log_level}")
        
        return errors
    
    @classmethod
    def for_environment(cls, env: str) -> "AppConfig":
        """Create configuration for specific environment."""
        base_config = {
            "environment": env,
            "database": DatabaseConfig.from_env(),
        }
        
        if env == "development":
            base_config.update({
                "debug": True,
                "secret_key": "dev-secret",
                "log_level": "DEBUG"
            })
        elif env == "testing":
            base_config.update({
                "debug": True,
                "secret_key": "test-secret",
                "database": DatabaseConfig(name="test_db"),
                "log_level": "DEBUG"
            })
        elif env == "production":
            base_config.update({
                "debug": False,
                "secret_key": os.getenv("SECRET_KEY", ""),
                "log_level": "INFO"
            })
        
        return cls(**base_config)

class ConfigurationManager:
    """Manage application configuration from multiple sources."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache: dict[str, Any] = {}
    
    def load_from_file(self, filename: str) -> dict[str, Any]:
        """Load configuration from file (JSON or YAML)."""
        if filename in self._cache:
            return self._cache[filename]
        
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if filename.endswith('.json'):
                    config = json.load(f)
                elif filename.endswith(('.yml', '.yaml')):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported configuration format: {filename}")
            
            self._cache[filename] = config
            return config
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Error parsing configuration file {filename}: {e}")
    
    def load_from_env(self, prefix: str = "APP_") -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Convert to appropriate type
                if value.lower() in ('true', 'false'):
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                else:
                    try:
                        config[config_key] = float(value)
                    except ValueError:
                        config[config_key] = value
        
        return config
    
    def merge_configs(self, *sources: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple configuration sources with proper precedence."""
        merged = {}
        
        for source in sources:
            self._deep_merge(merged, source)
        
        return merged
    
    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Recursively merge dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate configuration and return list of errors."""
    errors = []
    required_fields = ["secret_key", "environment"]
    
    for field in required_fields:
        if field not in config or not config[field]:
            errors.append(f"Required field missing: {field}")
    
    # Validate database configuration
    if "database" in config:
        db_config = config["database"]
        if not isinstance(db_config, dict):
            errors.append("Database configuration must be a dictionary")
        else:
            required_db_fields = ["host", "port", "name"]
            for field in required_db_fields:
                if field not in db_config:
                    errors.append(f"Required database field missing: {field}")
    
    return errors

# =============================================================================
# TASK 3: Package Structure Design - SOLUTION
# =============================================================================

# Task 3.1: Model Layer
class BaseModel(ABC):
    """Base class for all data models."""
    
    def __init__(self, **kwargs):
        self.id: int | None = kwargs.get('id')
        self.created_at: datetime = kwargs.get('created_at', datetime.now())
        self.updated_at: datetime = kwargs.get('updated_at', datetime.now())
        
        # Validate on creation
        errors = self.validate()
        if errors:
            raise ValueError(f"Model validation failed: {', '.join(errors)}")
    
    @abstractmethod
    def validate(self) -> list[str]:
        """Validate model data and return errors."""
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, **kwargs) -> None:
        """Update model attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
        
        # Re-validate after update
        errors = self.validate()
        if errors:
            raise ValueError(f"Model validation failed after update: {', '.join(errors)}")

class UserRole(Enum):
    """User role enumeration."""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    GUEST = "guest"

class User(BaseModel):
    """User model with comprehensive validation."""
    
    def __init__(self, username: str, email: str, **kwargs):
        self.username = username
        self.email = email
        self.role = UserRole(kwargs.get('role', UserRole.USER.value))
        self.is_active = kwargs.get('is_active', True)
        self.first_name = kwargs.get('first_name', '')
        self.last_name = kwargs.get('last_name', '')
        super().__init__(**kwargs)
    
    def validate(self) -> list[str]:
        """Validate user data."""
        errors = []
        
        if not self.username or len(self.username) < 3:
            errors.append("Username must be at least 3 characters long")
        
        if not self.email or '@' not in self.email:
            errors.append("Valid email address is required")
        
        if not isinstance(self.role, UserRole):
            errors.append("Invalid user role")
        
        return errors
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}".strip() or self.username
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        permissions = {
            UserRole.ADMIN: ["read", "write", "delete", "admin"],
            UserRole.MANAGER: ["read", "write", "delete"],
            UserRole.USER: ["read", "write"],
            UserRole.GUEST: ["read"]
        }
        return permission in permissions.get(self.role, [])

class ProductCategory(Enum):
    """Product category enumeration."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME = "home"
    SPORTS = "sports"

class Product(BaseModel):
    """Product model with pricing and inventory."""
    
    def __init__(self, name: str, price: float, **kwargs):
        self.name = name
        self.price = price
        self.category = ProductCategory(kwargs.get('category', ProductCategory.ELECTRONICS.value))
        self.description = kwargs.get('description', '')
        self.sku = kwargs.get('sku', '')
        self.inventory_count = kwargs.get('inventory_count', 0)
        self.is_active = kwargs.get('is_active', True)
        super().__init__(**kwargs)
    
    def validate(self) -> list[str]:
        """Validate product data."""
        errors = []
        
        if not self.name or len(self.name) < 2:
            errors.append("Product name must be at least 2 characters long")
        
        if self.price < 0:
            errors.append("Product price cannot be negative")
        
        if self.inventory_count < 0:
            errors.append("Inventory count cannot be negative")
        
        return errors
    
    def apply_discount(self, percentage: float) -> float:
        """Apply discount and return new price."""
        if not 0 <= percentage <= 100:
            raise ValueError("Discount percentage must be between 0 and 100")
        
        discount_amount = self.price * (percentage / 100)
        return self.price - discount_amount
    
    def is_in_stock(self) -> bool:
        """Check if product is in stock."""
        return self.inventory_count > 0 and self.is_active

# Task 3.2: Service Layer
class BaseService:
    """Base class for all services with dependency injection."""
    
    def __init__(self, repository=None, logger=None):
        self.repository = repository
        self.logger = logger or logging.getLogger(self.__class__.__name__)

class UserService(BaseService):
    """Service for user-related business logic."""
    
    def create_user(self, user_data: dict[str, Any]) -> User:
        """Create a new user with validation and business rules."""
        # Check if username already exists
        if self.repository:
            existing_user = self.repository.find_by_username(user_data.get('username'))
            if existing_user:
                raise ValueError("Username already exists")
        
        # Create user
        user = User(**user_data)
        
        # Save to repository
        if self.repository:
            user = self.repository.save(user)
        
        self.logger.info(f"Created user: {user.username}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user credentials."""
        if not self.repository:
            raise RuntimeError("Repository not configured")
        
        user = self.repository.find_by_username(username)
        if not user or not user.is_active:
            return None
        
        # In real implementation, check hashed password
        # For demo, we'll assume authentication succeeds
        self.logger.info(f"User authenticated: {username}")
        return user
    
    def update_user_role(self, user_id: int, new_role: UserRole) -> User:
        """Update user role with permission check."""
        if not self.repository:
            raise RuntimeError("Repository not configured")
        
        user = self.repository.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        user.role = new_role
        user = self.repository.save(user)
        
        self.logger.info(f"Updated user role: {user.username} -> {new_role.value}")
        return user

class ProductService(BaseService):
    """Service for product-related business logic."""
    
    def create_product(self, product_data: dict[str, Any]) -> Product:
        """Create a new product with business rules."""
        # Generate SKU if not provided
        if 'sku' not in product_data:
            product_data['sku'] = self._generate_sku(product_data.get('name', ''))
        
        product = Product(**product_data)
        
        if self.repository:
            product = self.repository.save(product)
        
        self.logger.info(f"Created product: {product.name}")
        return product
    
    def calculate_discount(self, product: Product, discount_percent: float) -> float:
        """Calculate discounted price with business rules."""
        if discount_percent > 50:
            self.logger.warning(f"Large discount applied: {discount_percent}% on {product.name}")
        
        return product.apply_discount(discount_percent)
    
    def update_inventory(self, product_id: int, quantity_change: int) -> Product:
        """Update product inventory."""
        if not self.repository:
            raise RuntimeError("Repository not configured")
        
        product = self.repository.find_by_id(product_id)
        if not product:
            raise ValueError("Product not found")
        
        new_inventory = product.inventory_count + quantity_change
        if new_inventory < 0:
            raise ValueError("Insufficient inventory")
        
        product.inventory_count = new_inventory
        product = self.repository.save(product)
        
        self.logger.info(f"Updated inventory for {product.name}: {product.inventory_count}")
        return product
    
    def _generate_sku(self, name: str) -> str:
        """Generate SKU from product name."""
        import hashlib
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8].upper()
        return f"PRD-{name_hash}"

# Task 3.3: Repository Pattern
T = TypeVar('T', bound=BaseModel)

class Repository(ABC):
    """Abstract repository for data access."""
    
    @abstractmethod
    def save(self, entity: T) -> T:
        """Save entity to storage."""
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: int) -> T | None:
        """Find entity by ID."""
        pass
    
    @abstractmethod
    def find_all(self) -> list[T]:
        """Find all entities."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: int) -> bool:
        """Delete entity by ID."""
        pass

class InMemoryRepository(Repository):
    """In-memory repository implementation."""
    
    def __init__(self):
        self._storage: dict[int, BaseModel] = {}
        self._next_id = 1
        self._lock = threading.Lock()
    
    def save(self, entity: BaseModel) -> BaseModel:
        """Save entity to memory storage."""
        with self._lock:
            if entity.id is None:
                entity.id = self._next_id
                self._next_id += 1
            
            entity.updated_at = datetime.now()
            self._storage[entity.id] = entity
            return entity
    
    def find_by_id(self, entity_id: int) -> BaseModel | None:
        """Find entity by ID."""
        return self._storage.get(entity_id)
    
    def find_all(self) -> list[BaseModel]:
        """Find all entities."""
        return list(self._storage.values())
    
    def delete(self, entity_id: int) -> bool:
        """Delete entity by ID."""
        with self._lock:
            if entity_id in self._storage:
                del self._storage[entity_id]
                return True
            return False
    
    def find_by_username(self, username: str) -> User | None:
        """Find user by username."""
        for entity in self._storage.values():
            if isinstance(entity, User) and entity.username == username:
                return entity
        return None
    
    def find_by_category(self, category: ProductCategory) -> list[Product]:
        """Find products by category."""
        return [
            entity for entity in self._storage.values()
            if isinstance(entity, Product) and entity.category == category
        ]

# =============================================================================
# TASK 4: Plugin Architecture - SOLUTION
# =============================================================================

class PluginState(Enum):
    """Plugin state enumeration."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"

class Plugin(ABC):
    """Base class for all plugins with lifecycle management."""
    
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    dependencies: list[str] = []
    
    def __init__(self):
        self.state = PluginState.UNLOADED
        self.config: dict[str, Any] = {}
        self._error_message: str | None = None
    
    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the plugin's main functionality."""
        pass
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self.state = PluginState.UNLOADED
        self.config.clear()
    
    def get_info(self) -> dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": self.dependencies,
            "state": self.state.value,
            "error": self._error_message
        }

class PluginManager:
    """Manage and coordinate plugins with dependency resolution."""
    
    def __init__(self):
        self._plugins: dict[str, Plugin] = {}
        self._load_order: list[str] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin."""
        if not plugin.name:
            raise ValueError("Plugin must have a name")
        
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin {plugin.name} already registered")
        
        self._plugins[plugin.name] = plugin
        self.logger.info(f"Registered plugin: {plugin.name}")
    
    def load_plugin(self, name: str, config: dict[str, Any] = None) -> None:
        """Load and initialize a plugin with dependency resolution."""
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} not registered")
        
        plugin = self._plugins[name]
        
        if plugin.state == PluginState.ACTIVE:
            self.logger.warning(f"Plugin {name} already loaded")
            return
        
        # Load dependencies first
        for dep_name in plugin.dependencies:
            if dep_name not in self._plugins:
                raise ValueError(f"Plugin {name} depends on {dep_name} which is not registered")
            
            dep_plugin = self._plugins[dep_name]
            if dep_plugin.state != PluginState.ACTIVE:
                self.load_plugin(dep_name)
        
        try:
            plugin.initialize(config or {})
            plugin.state = PluginState.ACTIVE
            plugin.config = config or {}
            
            if name not in self._load_order:
                self._load_order.append(name)
            
            self.logger.info(f"Loaded plugin: {name}")
            
        except Exception as e:
            plugin.state = PluginState.ERROR
            plugin._error_message = str(e)
            self.logger.error(f"Failed to load plugin {name}: {e}")
            raise
    
    def unload_plugin(self, name: str) -> None:
        """Unload a plugin and its dependents."""
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} not registered")
        
        # Find and unload dependent plugins first
        dependents = self._find_dependents(name)
        for dependent in dependents:
            if self._plugins[dependent].state == PluginState.ACTIVE:
                self.unload_plugin(dependent)
        
        plugin = self._plugins[name]
        if plugin.state == PluginState.ACTIVE:
            try:
                plugin.cleanup()
                if name in self._load_order:
                    self._load_order.remove(name)
                self.logger.info(f"Unloaded plugin: {name}")
            except Exception as e:
                self.logger.error(f"Error unloading plugin {name}: {e}")
    
    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        """Execute a plugin by name."""
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} not found")
        
        plugin = self._plugins[name]
        
        if plugin.state != PluginState.ACTIVE:
            raise RuntimeError(f"Plugin {name} is not active (state: {plugin.state.value})")
        
        try:
            return plugin.execute(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error executing plugin {name}: {e}")
            raise
    
    def list_plugins(self) -> list[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
    
    def get_plugin_info(self, name: str) -> dict[str, Any]:
        """Get information about a specific plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} not found")
        
        return self._plugins[name].get_info()
    
    def _find_dependents(self, plugin_name: str) -> list[str]:
        """Find plugins that depend on the given plugin."""
        dependents = []
        for name, plugin in self._plugins.items():
            if plugin_name in plugin.dependencies:
                dependents.append(name)
        return dependents

# Task 4.3: Example Plugins
class EmailPlugin(Plugin):
    """Plugin for sending emails via SMTP."""
    
    name = "email"
    version = "1.0.0"
    description = "Send emails via SMTP"
    
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize email configuration."""
        self.smtp_server = config.get("smtp_server", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.use_tls = config.get("use_tls", True)
        
        self.state = PluginState.LOADED
    
    def execute(self, to: str, subject: str, body: str, **kwargs) -> bool:
        """Send an email."""
        # Simulate email sending
        print(f"Sending email to {to}")
        print(f"Subject: {subject}")
        print(f"Body: {body}")
        print(f"SMTP Server: {self.smtp_server}:{self.smtp_port}")
        
        # In real implementation, use smtplib
        return True

class LoggingPlugin(Plugin):
    """Plugin for enhanced logging functionality."""
    
    name = "logging"
    version = "1.0.0" 
    description = "Enhanced logging functionality"
    
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize logging configuration."""
        self.log_level = config.get("log_level", "INFO")
        self.log_format = config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.log_file = config.get("log_file")
        
        # Configure logger
        self.logger = logging.getLogger("enhanced_logger")
        self.logger.setLevel(getattr(logging, self.log_level))
        
        formatter = logging.Formatter(self.log_format)
        
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.state = PluginState.LOADED
    
    def execute(self, level: str, message: str, **kwargs) -> None:
        """Log a message with enhanced formatting."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        
        # Add context information
        context = kwargs.get("context", {})
        if context:
            message = f"{message} | Context: {context}"
        
        log_method(message)

class CachePlugin(Plugin):
    """Plugin for caching functionality."""
    
    name = "cache"
    version = "1.0.0"
    description = "Simple in-memory caching"
    dependencies = ["logging"]
    
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize cache configuration."""
        self.max_size = config.get("max_size", 1000)
        self.ttl = config.get("ttl", 3600)  # Time to live in seconds
        self._cache: dict[str, dict[str, Any]] = {}
        
        self.state = PluginState.LOADED
    
    def execute(self, action: str, key: str = None, value: Any = None, **kwargs) -> Any:
        """Execute cache operations."""
        if action == "get":
            return self._get(key)
        elif action == "set":
            return self._set(key, value)
        elif action == "delete":
            return self._delete(key)
        elif action == "clear":
            return self._clear()
        else:
            raise ValueError(f"Unknown cache action: {action}")
    
    def _get(self, key: str) -> Any:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return None
        
        return entry["value"]
    
    def _set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = {
            "value": value,
            "timestamp": datetime.now()
        }
    
    def _delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def _clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.now() - entry["timestamp"]).total_seconds()
        return age > self.ttl

# =============================================================================
# TASK 5: Dependency Injection Container - SOLUTION
# =============================================================================

class ServiceLifetime(Enum):
    """Service lifetime enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

@dataclass
class ServiceRegistration:
    """Service registration information."""
    interface: Type
    implementation: Type
    lifetime: ServiceLifetime
    instance: Any = None
    factory: Callable | None = None

class ServiceContainer:
    """Container for dependency injection with advanced features."""
    
    def __init__(self):
        self._registrations: dict[Type, ServiceRegistration] = {}
        self._singletons: dict[Type, Any] = {}
        self._scoped_instances: dict[Type, Any] = {}
        self._resolving: set = set()  # Circular dependency detection
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_singleton(self, interface: Type, implementation: Type = None) -> None:
        """Register a singleton service."""
        implementation = implementation or interface
        self._registrations[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON
        )
        self.logger.debug(f"Registered singleton: {interface.__name__} -> {implementation.__name__}")
    
    def register_transient(self, interface: Type, implementation: Type = None) -> None:
        """Register a transient service."""
        implementation = implementation or interface
        self._registrations[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT
        )
        self.logger.debug(f"Registered transient: {interface.__name__} -> {implementation.__name__}")
    
    def register_scoped(self, interface: Type, implementation: Type = None) -> None:
        """Register a scoped service."""
        implementation = implementation or interface
        self._registrations[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED
        )
        self.logger.debug(f"Registered scoped: {interface.__name__} -> {implementation.__name__}")
    
    def register_instance(self, interface: Type, instance: Any) -> None:
        """Register a specific instance."""
        self._registrations[interface] = ServiceRegistration(
            interface=interface,
            implementation=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            instance=instance
        )
        self._singletons[interface] = instance
        self.logger.debug(f"Registered instance: {interface.__name__}")
    
    def register_factory(self, interface: Type, factory: Callable, lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> None:
        """Register a factory function."""
        self._registrations[interface] = ServiceRegistration(
            interface=interface,
            implementation=None,
            lifetime=lifetime,
            factory=factory
        )
        self.logger.debug(f"Registered factory: {interface.__name__}")
    
    def resolve(self, service_type: Type) -> Any:
        """Resolve a service and its dependencies."""
        if service_type in self._resolving:
            raise RuntimeError(f"Circular dependency detected: {service_type.__name__}")
        
        if service_type not in self._registrations:
            raise ValueError(f"Service not registered: {service_type.__name__}")
        
        registration = self._registrations[service_type]
        
        # Handle singleton
        if registration.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            instance = self._create_instance(registration)
            self._singletons[service_type] = instance
            return instance
        
        # Handle scoped
        elif registration.lifetime == ServiceLifetime.SCOPED:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            instance = self._create_instance(registration)
            self._scoped_instances[service_type] = instance
            return instance
        
        # Handle transient
        else:
            return self._create_instance(registration)
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create an instance using registration information."""
        try:
            self._resolving.add(registration.interface)
            
            # Use existing instance
            if registration.instance is not None:
                return registration.instance
            
            # Use factory
            if registration.factory is not None:
                return registration.factory(self)
            
            # Create from implementation
            implementation = registration.implementation
            
            # Get constructor parameters
            sig = inspect.signature(implementation.__init__)
            dependencies = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                if param.annotation == param.empty:
                    continue
                
                # Resolve dependency
                dependency = self.resolve(param.annotation)
                dependencies[param_name] = dependency
            
            return implementation(**dependencies)
            
        finally:
            self._resolving.discard(registration.interface)
    
    def clear_scoped(self) -> None:
        """Clear scoped instances."""
        self._scoped_instances.clear()
    
    def get_registrations(self) -> dict[Type, ServiceRegistration]:
        """Get all service registrations."""
        return self._registrations.copy()

# Task 5.2: Service Decorators
_global_container = ServiceContainer()

def singleton(interface: Type = None):
    """Decorator to register a class as singleton."""
    def decorator(cls):
        target_interface = interface or cls
        _global_container.register_singleton(target_interface, cls)
        return cls
    return decorator

def transient(interface: Type = None):
    """Decorator to register a class as transient.""" 
    def decorator(cls):
        target_interface = interface or cls
        _global_container.register_transient(target_interface, cls)
        return cls
    return decorator

def scoped(interface: Type = None):
    """Decorator to register a class as scoped."""
    def decorator(cls):
        target_interface = interface or cls
        _global_container.register_scoped(target_interface, cls)
        return cls
    return decorator

# Task 5.3: Example Services with Dependencies
class DatabaseConnection:
    """Example database connection service."""
    
    def __init__(self, connection_string: str = "sqlite:///:memory:"):
        self.connection_string = connection_string
        self.is_connected = False
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def connect(self) -> None:
        """Connect to database."""
        self.is_connected = True
        self.logger.info(f"Connected to database: {self.connection_string}")
    
    def disconnect(self) -> None:
        """Disconnect from database."""
        self.is_connected = False
        self.logger.info("Disconnected from database")
    
    def execute(self, query: str) -> list[dict[str, Any]]:
        """Execute database query."""
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
        
        # Simulate query execution
        self.logger.debug(f"Executing query: {query}")
        return [{"id": 1, "result": "data"}]

class UserRepository:
    """User repository with database dependency."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save(self, user: User) -> User:
        """Save user to database."""
        query = f"INSERT INTO users (username, email) VALUES ('{user.username}', '{user.email}')"
        self.db_connection.execute(query)
        self.logger.info(f"Saved user: {user.username}")
        return user
    
    def find_by_username(self, username: str) -> User | None:
        """Find user by username."""
        query = f"SELECT * FROM users WHERE username = '{username}'"
        results = self.db_connection.execute(query)
        
        if results:
            # Simulate user creation from database result
            return User(username=username, email="user@example.com")
        
        return None

class NotificationService:
    """Notification service with multiple dependencies."""
    
    def __init__(self, user_repository: UserRepository, logger: logging.Logger = None):
        self.user_repository = user_repository
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def send_welcome_email(self, username: str) -> bool:
        """Send welcome email to user."""
        user = self.user_repository.find_by_username(username)
        if not user:
            self.logger.error(f"User not found: {username}")
            return False
        
        # Simulate sending email
        self.logger.info(f"Sending welcome email to {user.email}")
        return True

# =============================================================================
# TASK 6: Testing Structure - SOLUTION
# =============================================================================

# Task 6.1: Test Fixtures
class TestFixtures:
    """Reusable test fixtures with comprehensive defaults."""
    
    @staticmethod
    def create_test_user(**kwargs) -> User:
        """Create a test user with sensible defaults."""
        defaults = {
            "username": "testuser",
            "email": "test@example.com",
            "role": UserRole.USER.value,
            "is_active": True,
            "first_name": "Test",
            "last_name": "User"
        }
        defaults.update(kwargs)
        return User(**defaults)
    
    @staticmethod
    def create_test_product(**kwargs) -> Product:
        """Create a test product with sensible defaults."""
        defaults = {
            "name": "Test Product",
            "price": 99.99,
            "category": ProductCategory.ELECTRONICS.value,
            "description": "A test product for testing purposes",
            "sku": "TEST-001",
            "inventory_count": 10,
            "is_active": True
        }
        defaults.update(kwargs)
        return Product(**defaults)
    
    @staticmethod
    def create_test_config(**kwargs) -> AppConfig:
        """Create test configuration."""
        defaults = {
            "debug": True,
            "secret_key": "test-secret",
            "environment": "testing",
            "log_level": "DEBUG"
        }
        defaults.update(kwargs)
        return AppConfig(**defaults)
    
    @staticmethod
    def create_test_database_config(**kwargs) -> DatabaseConfig:
        """Create test database configuration."""
        defaults = {
            "host": "localhost",
            "port": 5432,
            "name": "test_db",
            "user": "test_user",
            "password": "test_pass"
        }
        defaults.update(kwargs)
        return DatabaseConfig(**defaults)

# Task 6.2: Test Base Classes
class BaseTestCase:
    """Base class for all test cases with common setup/teardown."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clear any global state
        _global_container._singletons.clear()
        _global_container._scoped_instances.clear()
        
        # Set up test logging
        logging.basicConfig(level=logging.DEBUG)
        
        # Create test fixtures
        self.test_user = TestFixtures.create_test_user()
        self.test_product = TestFixtures.create_test_product()
        self.test_config = TestFixtures.create_test_config()
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clear any test data
        pass
    
    def assert_valid_model(self, model: BaseModel) -> None:
        """Assert that a model is valid."""
        errors = model.validate()
        assert not errors, f"Model validation failed: {errors}"
    
    def assert_model_equals(self, model1: BaseModel, model2: BaseModel) -> None:
        """Assert that two models are equal."""
        assert model1.to_dict() == model2.to_dict()

class IntegrationTestCase(BaseTestCase):
    """Base class for integration tests with database setup."""
    
    def setup_method(self):
        """Set up integration test environment."""
        super().setup_method()
        
        # Set up test database
        self.db_connection = DatabaseConnection("sqlite:///:memory:")
        self.db_connection.connect()
        
        # Set up repositories
        self.user_repository = InMemoryRepository()
        self.product_repository = InMemoryRepository()
        
        # Set up services
        self.user_service = UserService(self.user_repository)
        self.product_service = ProductService(self.product_repository)
    
    def teardown_method(self):
        """Clean up integration test environment."""
        super().teardown_method()
        
        if hasattr(self, 'db_connection'):
            self.db_connection.disconnect()

# Task 6.3: Test Utilities
class TestDatabase:
    """Test database utilities for managing test data."""
    
    @staticmethod
    def create_test_database() -> DatabaseConnection:
        """Create an in-memory test database."""
        db = DatabaseConnection("sqlite:///:memory:")
        db.connect()
        
        # Create test tables
        TestDatabase._create_test_tables(db)
        return db
    
    @staticmethod
    def _create_test_tables(db: DatabaseConnection) -> None:
        """Create test database tables."""
        # In a real implementation, this would create actual tables
        db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, email TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
    
    @staticmethod
    def clear_database(db: DatabaseConnection) -> None:
        """Clear all data from test database."""
        db.execute("DELETE FROM users")
        db.execute("DELETE FROM products")
    
    @staticmethod
    def seed_test_data(db: DatabaseConnection) -> None:
        """Seed database with test data."""
        # Insert test users
        db.execute("INSERT INTO users (username, email) VALUES ('user1', 'user1@test.com')")
        db.execute("INSERT INTO users (username, email) VALUES ('user2', 'user2@test.com')")
        
        # Insert test products
        db.execute("INSERT INTO products (name, price) VALUES ('Product 1', 19.99)")
        db.execute("INSERT INTO products (name, price) VALUES ('Product 2', 29.99)")

class TestHelpers:
    """General test helper functions and utilities."""
    
    @staticmethod
    def assert_valid_email(email: str) -> bool:
        """Assert that email format is valid."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        is_valid = bool(re.match(email_pattern, email))
        assert is_valid, f"Invalid email format: {email}"
        return is_valid
    
    @staticmethod
    def generate_test_data(count: int, data_type: str) -> list[Any]:
        """Generate test data of specified type."""
        if data_type == "users":
            return [
                TestFixtures.create_test_user(
                    username=f"user{i}",
                    email=f"user{i}@test.com"
                )
                for i in range(count)
            ]
        elif data_type == "products":
            return [
                TestFixtures.create_test_product(
                    name=f"Product {i}",
                    price=10.0 + i,
                    sku=f"PROD-{i:03d}"
                )
                for i in range(count)
            ]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    @staticmethod
    def create_mock_repository() -> InMemoryRepository:
        """Create a mock repository with test data."""
        repo = InMemoryRepository()
        
        # Add some test data
        users = TestHelpers.generate_test_data(3, "users")
        products = TestHelpers.generate_test_data(5, "products")
        
        for user in users:
            repo.save(user)
        
        for product in products:
            repo.save(product)
        
        return repo
    
    @staticmethod
    def assert_performance(func: Callable, max_time: float) -> Any:
        """Assert that function executes within time limit."""
        import time
        
        start_time = time.time()
        result = func()
        duration = time.time() - start_time
        
        assert duration <= max_time, f"Function took {duration:.3f}s, expected <= {max_time}s"
        return result

# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

def demonstrate_all_solutions():
    """Demonstrate all implemented project structure patterns."""
    print("🏗️  Project Structure Workshop - Complete Solutions")
    print("=" * 60)
    
    # Task 1: Calculator Module Organization
    print("\n📁 TASK 1: Module Organization")
    print("-" * 40)
    
    calc = Calculator()
    result1 = calc.add(10, 5)
    result2 = calc.multiply(3, 4)
    result3 = calc.sqrt(16)
    
    print(f"Calculator operations:")
    print(f"  10 + 5 = {result1}")
    print(f"  3 * 4 = {result2}")
    print(f"  sqrt(16) = {result3}")
    print(f"  Constants: PI = {MathConstants.PI:.4f}")
    print(f"  History: {len(calc.get_history())} operations")
    
    # Task 2: Configuration Management
    print("\n⚙️  TASK 2: Configuration Management")
    print("-" * 40)
    
    config_mgr = ConfigurationManager()
    
    # Create test config
    test_config = {
        "debug": True,
        "secret_key": "test-secret",
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db"
        }
    }
    
    app_config = AppConfig.for_environment("development")
    print(f"App config for development:")
    print(f"  Debug: {app_config.debug}")
    print(f"  Environment: {app_config.environment}")
    print(f"  Database host: {app_config.database.host}")
    
    # Task 3: Package Structure
    print("\n📦 TASK 3: Package Structure")
    print("-" * 40)
    
    # Create repository and services
    user_repo = InMemoryRepository()
    user_service = UserService(user_repo)
    
    # Create test user
    user_data = {
        "username": "john_doe",
        "email": "john@example.com",
        "role": UserRole.USER.value
    }
    
    user = user_service.create_user(user_data)
    print(f"Created user: {user.username} ({user.email})")
    print(f"User permissions - can read: {user.has_permission('read')}")
    print(f"User permissions - can admin: {user.has_permission('admin')}")
    
    # Create test product
    product_repo = InMemoryRepository()
    product_service = ProductService(product_repo)
    
    product_data = {
        "name": "Laptop",
        "price": 999.99,
        "category": ProductCategory.ELECTRONICS.value,
        "inventory_count": 5
    }
    
    product = product_service.create_product(product_data)
    discounted_price = product_service.calculate_discount(product, 10)
    print(f"Created product: {product.name} (${product.price})")
    print(f"With 10% discount: ${discounted_price:.2f}")
    
    # Task 4: Plugin Architecture
    print("\n🔌 TASK 4: Plugin Architecture")
    print("-" * 40)
    
    plugin_mgr = PluginManager()
    
    # Register plugins
    email_plugin = EmailPlugin()
    logging_plugin = LoggingPlugin()
    cache_plugin = CachePlugin()
    
    plugin_mgr.register_plugin(email_plugin)
    plugin_mgr.register_plugin(logging_plugin)
    plugin_mgr.register_plugin(cache_plugin)
    
    # Load plugins
    plugin_mgr.load_plugin("logging", {"log_level": "INFO"})
    plugin_mgr.load_plugin("email", {"smtp_server": "smtp.example.com"})
    plugin_mgr.load_plugin("cache", {"max_size": 100})
    
    # Execute plugins
    plugin_mgr.execute_plugin("email", "user@example.com", "Welcome!", "Welcome to our service!")
    plugin_mgr.execute_plugin("cache", "set", "user:1", {"name": "John", "email": "john@example.com"})
    cached_user = plugin_mgr.execute_plugin("cache", "get", "user:1")
    
    print(f"Loaded plugins: {plugin_mgr.list_plugins()}")
    print(f"Cached user data: {cached_user}")
    
    # Task 5: Dependency Injection
    print("\n💉 TASK 5: Dependency Injection")
    print("-" * 40)
    
    container = ServiceContainer()
    
    # Register services
    container.register_singleton(DatabaseConnection)
    container.register_transient(UserRepository)
    container.register_scoped(NotificationService)
    
    # Resolve services
    db_conn = container.resolve(DatabaseConnection)
    db_conn.connect()
    
    notification_service = container.resolve(NotificationService)
    
    print(f"Database connected: {db_conn.is_connected}")
    print(f"Service registrations: {len(container.get_registrations())}")
    
    # Task 6: Testing Structure
    print("\n🧪 TASK 6: Testing Structure")
    print("-" * 40)
    
    # Create test fixtures
    test_user = TestFixtures.create_test_user(username="test_user")
    test_product = TestFixtures.create_test_product(name="Test Product")
    
    # Generate test data
    test_users = TestHelpers.generate_test_data(3, "users")
    test_products = TestHelpers.generate_test_data(2, "products")
    
    print(f"Test user created: {test_user.username}")
    print(f"Test product created: {test_product.name}")
    print(f"Generated {len(test_users)} test users")
    print(f"Generated {len(test_products)} test products")
    
    # Validate email
    TestHelpers.assert_valid_email("test@example.com")
    print("Email validation passed")
    
    print("\n🎉 All project structure patterns demonstrated successfully!")
    print("   ✅ Module organization with calculator package")
    print("   ✅ Configuration management with multiple sources")
    print("   ✅ Layered architecture with models, services, repositories")
    print("   ✅ Plugin system with dependency resolution")
    print("   ✅ Dependency injection container with lifetimes")
    print("   ✅ Testing infrastructure with fixtures and utilities")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    demonstrate_all_solutions()
