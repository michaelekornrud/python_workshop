"""
Testing Workshop - Practice Exercises

Complete the following tasks to practice testing concepts and techniques.
Apply the principles from the testing.md file to write comprehensive test suites.
"""

import pytest
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# =============================================================================
# TASK 1: Basic Unit Testing with pytest
# =============================================================================

"""
TASK 1: Create a Calculator Class and Test Suite

Create a Calculator class with basic mathematical operations and comprehensive tests.

Requirements:
- Class name: Calculator
- Methods: add, subtract, multiply, divide, power
- Handle division by zero with appropriate exception
- Include memory functionality (store, recall, clear)
- Write comprehensive test suite with proper assertions
- Test both success and error cases

Example usage:
calc = Calculator()
calc.add(2, 3) -> 5
calc.divide(10, 0) -> raises ValueError
calc.store(42) -> stores value in memory
calc.recall() -> returns stored value
"""

class Calculator:
    """Your Calculator implementation here"""
    pass

class TestCalculator:
    """Your test suite here"""
    pass

# =============================================================================
# TASK 2: Fixtures and Test Data Management
# =============================================================================

"""
TASK 2: User Management System with Fixtures

Create a UserManager class and comprehensive test suite using pytest fixtures.

Requirements:
- Class name: UserManager
- Methods: create_user, get_user, update_user, delete_user, list_users
- User should have: id, name, email, created_at, is_active
- Use fixtures for: empty manager, populated manager, sample users
- Test user lifecycle operations
- Use proper fixture scopes and dependencies

Example usage:
manager = UserManager()
user = manager.create_user("Alice", "alice@example.com")
manager.update_user(user.id, name="Alice Smith")
manager.delete_user(user.id)
"""

class User:
    """Your User class here"""
    pass

class UserManager:
    """Your UserManager implementation here"""
    pass

# Fixtures to implement
@pytest.fixture
def user_manager():
    """Your fixture here"""
    pass

@pytest.fixture
def sample_users():
    """Your fixture here"""
    pass

@pytest.fixture
def populated_manager(user_manager, sample_users):
    """Your fixture here"""
    pass

class TestUserManager:
    """Your test suite using fixtures"""
    pass

# =============================================================================
# TASK 3: Parameterized Testing
# =============================================================================

"""
TASK 3: Input Validation with Parameterized Tests

Create validation functions and test them with multiple input scenarios.

Requirements:
- Function: validate_email(email) -> bool
- Function: validate_password(password) -> ValidationResult
- Function: validate_age(age) -> bool
- Password requirements: 8+ chars, uppercase, lowercase, digit, special char
- Age requirements: 0-150 range
- Use @pytest.mark.parametrize for comprehensive testing
- Test edge cases and boundary values

Example usage:
validate_email("user@example.com") -> True
validate_password("Weak123") -> ValidationResult(valid=False, errors=["no_special"])
validate_age(25) -> True
"""

class ValidationResult:
    """Your ValidationResult class here"""
    pass

def validate_email(email: str) -> bool:
    """Your email validation function here"""
    pass

def validate_password(password: str) -> ValidationResult:
    """Your password validation function here"""
    pass

def validate_age(age: int) -> bool:
    """Your age validation function here"""
    pass

class TestValidation:
    """Your parameterized test suite here"""
    pass

# =============================================================================
# TASK 4: Mocking External Dependencies
# =============================================================================

"""
TASK 4: Email Service with Mocked Dependencies

Create an EmailService that sends emails through an external provider and test it with mocks.

Requirements:
- Class name: EmailService
- Methods: send_email, send_welcome_email, send_password_reset
- Use external EmailProvider (to be mocked)
- Handle provider failures gracefully
- Include retry mechanism for failed sends
- Mock the EmailProvider in tests
- Test both success and failure scenarios

Example usage:
service = EmailService(provider)
service.send_email("user@example.com", "Subject", "Body")
service.send_welcome_email(user)
service.send_password_reset(user, reset_token)
"""

class EmailProvider:
    """External email provider interface (to be mocked)"""
    def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        """Send email and return result"""
        pass

class EmailService:
    """Your EmailService implementation here"""
    pass

class TestEmailService:
    """Your test suite with mocking"""
    pass

# =============================================================================
# TASK 5: Property-Based Testing with Hypothesis
# =============================================================================

"""
TASK 5: Mathematical Functions with Property-Based Tests

Create mathematical utility functions and test them with Hypothesis.

Requirements:
- Function: factorial(n) -> int
- Function: fibonacci(n) -> int
- Function: is_prime(n) -> bool
- Function: gcd(a, b) -> int (greatest common divisor)
- Use Hypothesis to generate test cases
- Test mathematical properties and invariants
- Handle edge cases (negative numbers, zero, large values)

Example usage:
factorial(5) -> 120
fibonacci(10) -> 55
is_prime(17) -> True
gcd(48, 18) -> 6
"""

def factorial(n: int) -> int:
    """Your factorial implementation here"""
    pass

def fibonacci(n: int) -> int:
    """Your fibonacci implementation here"""
    pass

def is_prime(n: int) -> bool:
    """Your prime checking implementation here"""
    pass

def gcd(a: int, b: int) -> int:
    """Your GCD implementation here"""
    pass

class TestMathematicalProperties:
    """Your property-based test suite here"""
    pass

# =============================================================================
# TASK 6: Integration Testing
# =============================================================================

"""
TASK 6: Blog System Integration Tests

Create a simple blog system and test component interactions.

Requirements:
- Classes: Blog, Post, Author, CommentService
- Blog manages posts and authors
- CommentService handles post comments
- Posts have: id, title, content, author, created_at, published
- Authors have: id, name, email, posts
- Comments have: id, post_id, author_name, content, created_at
- Test complete workflows (create author, write post, add comments)
- Test component interactions

Example workflow:
blog = Blog()
author = blog.create_author("Alice", "alice@example.com")
post = blog.create_post(author.id, "Title", "Content")
blog.publish_post(post.id)
comment_service.add_comment(post.id, "Reader", "Great post!")
"""

class Author:
    """Your Author class here"""
    pass

class Post:
    """Your Post class here"""
    pass

class Comment:
    """Your Comment class here"""
    pass

class Blog:
    """Your Blog class here"""
    pass

class CommentService:
    """Your CommentService class here"""
    pass

class TestBlogIntegration:
    """Your integration test suite here"""
    pass

# =============================================================================
# TASK 7: Exception Testing and Error Handling
# =============================================================================

"""
TASK 7: File Processing with Exception Testing

Create a file processor that handles various file operations and test error scenarios.

Requirements:
- Class name: FileProcessor
- Methods: read_json, write_json, backup_file, process_batch
- Handle: FileNotFoundError, PermissionError, JSONDecodeError
- Custom exceptions: ProcessingError, BackupError
- Test all exception scenarios
- Test exception chaining and error messages

Example usage:
processor = FileProcessor()
data = processor.read_json("data.json")
processor.write_json("output.json", data)
processor.backup_file("important.json")
processor.process_batch(["file1.json", "file2.json"])
"""

class ProcessingError(Exception):
    """Your custom exception here"""
    pass

class BackupError(Exception):
    """Your custom exception here"""
    pass

class FileProcessor:
    """Your FileProcessor implementation here"""
    pass

class TestFileProcessor:
    """Your exception testing suite here"""
    pass

# =============================================================================
# TASK 8: Test Performance and Optimization
# =============================================================================

"""
TASK 8: Data Analyzer with Performance Tests

Create a data analysis tool and test its performance characteristics.

Requirements:
- Class name: DataAnalyzer
- Methods: load_data, calculate_statistics, find_outliers, generate_report
- Handle large datasets efficiently
- Statistics: mean, median, std_dev, min, max
- Outliers: values beyond 2 standard deviations
- Performance tests for large datasets (1000+ items)
- Memory usage tests

Example usage:
analyzer = DataAnalyzer()
analyzer.load_data([1, 2, 3, ..., 1000])
stats = analyzer.calculate_statistics()
outliers = analyzer.find_outliers()
report = analyzer.generate_report()
"""

class DataAnalyzer:
    """Your DataAnalyzer implementation here"""
    pass

class TestDataAnalyzerPerformance:
    """Your performance test suite here"""
    pass

# =============================================================================
# TASK 9: Test Organization and Best Practices
# =============================================================================

"""
TASK 9: Complete E-commerce Cart System

Create a comprehensive e-commerce cart system with well-organized tests.

Requirements:
- Classes: Product, Cart, Discount, ShippingCalculator
- Product: id, name, price, category, in_stock
- Cart: add_item, remove_item, update_quantity, calculate_total
- Discount: percentage and fixed amount discounts
- ShippingCalculator: calculate based on weight and distance
- Organize tests by functionality
- Use appropriate test markers (unit, integration, slow)
- Include test helpers and utilities

Example usage:
cart = Cart()
product = Product(1, "Laptop", 999.99, "Electronics", True)
cart.add_item(product, quantity=1)
discount = Discount("SAVE10", 0.10)
cart.apply_discount(discount)
shipping = ShippingCalculator().calculate(cart, distance=100)
total = cart.calculate_total(include_shipping=True)
"""

class Product:
    """Your Product class here"""
    pass

class Cart:
    """Your Cart class here"""
    pass

class Discount:
    """Your Discount class here"""
    pass

class ShippingCalculator:
    """Your ShippingCalculator class here"""
    pass

# Test organization examples
class TestProduct:
    """Unit tests for Product class"""
    pass

class TestCart:
    """Unit tests for Cart class"""
    pass

class TestEcommerceIntegration:
    """Integration tests for e-commerce workflow"""
    pass

# =============================================================================
# BONUS TASK: Advanced Testing Patterns
# =============================================================================

"""
BONUS TASK: Test Framework Extensions

Create custom pytest plugins and advanced testing utilities.

Requirements:
- Custom pytest fixture for database setup/teardown
- Custom assertion helpers for domain objects
- Test data factory functions
- Custom markers for test categorization
- Parametrized test generator functions
- Mock factory for complex objects

Example usage:
@pytest.mark.api_test
def test_api_endpoint(api_client, user_factory):
    user = user_factory.create_user()
    response = api_client.get(f"/users/{user.id}")
    assert_valid_user_response(response, user)
"""

# Custom fixtures and utilities
def create_test_database():
    """Your database fixture here"""
    pass

def user_factory():
    """Your user factory here"""
    pass

def assert_valid_user_response(response, expected_user):
    """Your custom assertion here"""
    pass

class TestFrameworkExtensions:
    """Your advanced testing patterns here"""
    pass

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# pytest.ini content (create this file in your project root):
"""
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --tb=short
    -v
markers =
    unit: marks tests as unit tests
    integration: marks tests as integration tests
    slow: marks tests as slow running
    api_test: marks tests as API tests
"""

if __name__ == "__main__":
    # This would run the tests when the file is executed
    pytest.main([__file__])
