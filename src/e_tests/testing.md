# Testing in Python

Testing is fundamental to building reliable software. Python's rich testing ecosystem provides tools for unit testing, integration testing, property-based testing, and test automation. Good testing practices help catch bugs early, enable confident refactoring, and serve as living documentation of how your code should behave.

## Testing Fundamentals: Building Confidence in Code

Testing is not just about finding bugs—it's about designing better code, documenting expected behavior, and enabling safe changes. Well-tested code is easier to maintain, refactor, and extend, making development more productive over time.

## 1. Unit Testing with pytest

pytest is Python's most popular testing framework, offering powerful features while maintaining simplicity. It provides clear syntax, excellent error reporting, and extensive plugin ecosystem that makes testing productive and enjoyable.

### Basic Test Structure and Assertions

Good tests are clear, focused, and follow the Arrange-Act-Assert pattern:

```python
# test_calculator.py - Basic pytest structure
import pytest
from calculator import Calculator

class TestCalculator:
    """Test suite for Calculator class."""
    
    def test_addition(self):
        """Test basic addition functionality."""
        # Arrange
        calc = Calculator()
        
        # Act
        result = calc.add(2, 3)
        
        # Assert
        assert result == 5
    
    def test_division_by_zero(self):
        """Test that division by zero raises appropriate error."""
        calc = Calculator()
        
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(10, 0)
    
    def test_multiple_operations(self):
        """Test chaining multiple operations."""
        calc = Calculator()
        
        result = calc.add(5, 3)
        result = calc.multiply(result, 2)
        result = calc.subtract(result, 4)
        
        assert result == 12
```

The pytest framework automatically discovers tests and provides detailed failure information when assertions fail.

### Fixtures: Managing Test Dependencies

Fixtures provide a clean way to set up test data and resources, ensuring tests are isolated and repeatable:

```python
# test_user_service.py - Advanced fixture usage
import pytest
from datetime import datetime
from user_service import UserService, User
from database import Database

@pytest.fixture
def database():
    """Create a test database instance."""
    db = Database(":memory:")  # In-memory SQLite for testing
    db.create_tables()
    yield db
    db.close()

@pytest.fixture
def user_service(database):
    """Create UserService with test database."""
    return UserService(database)

@pytest.fixture
def sample_users():
    """Create sample user data for testing."""
    return [
        {"name": "Alice Smith", "email": "alice@example.com", "age": 30},
        {"name": "Bob Jones", "email": "bob@example.com", "age": 25},
        {"name": "Charlie Brown", "email": "charlie@example.com", "age": 35}
    ]

@pytest.fixture
def populated_service(user_service, sample_users):
    """Create UserService with sample data."""
    for user_data in sample_users:
        user_service.create_user(**user_data)
    return user_service

class TestUserService:
    """Comprehensive tests for UserService."""
    
    def test_create_user(self, user_service):
        """Test user creation."""
        user = user_service.create_user("Test User", "test@example.com", 25)
        
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.age == 25
        assert isinstance(user.created_at, datetime)
    
    def test_get_user_by_email(self, populated_service):
        """Test retrieving user by email."""
        user = populated_service.get_user_by_email("alice@example.com")
        
        assert user is not None
        assert user.name == "Alice Smith"
        assert user.age == 30
    
    def test_get_users_by_age_range(self, populated_service):
        """Test filtering users by age range."""
        users = populated_service.get_users_by_age_range(25, 30)
        
        assert len(users) == 2
        assert all(25 <= user.age <= 30 for user in users)
```

Fixtures promote test isolation and make complex test setups reusable across multiple test functions.

## 2. Parameterized Testing and Test Generation

### Testing Multiple Scenarios Efficiently

Parameterized tests allow you to test the same logic with different inputs, reducing code duplication and ensuring comprehensive coverage:

```python
# test_validation.py - Parameterized testing examples
import pytest
from validation import validate_email, validate_password, validate_phone

class TestEmailValidation:
    """Test email validation with various inputs."""
    
    @pytest.mark.parametrize("email,expected", [
        ("user@example.com", True),
        ("test.user+tag@domain.co.uk", True),
        ("simple@domain.org", True),
        ("invalid.email", False),
        ("@domain.com", False),
        ("user@", False),
        ("", False),
        ("user@@domain.com", False),
    ])
    def test_email_validation(self, email, expected):
        """Test email validation with various formats."""
        result = validate_email(email)
        assert result == expected
    
    @pytest.mark.parametrize("password,expected_valid,expected_errors", [
        ("ValidPass123!", True, []),
        ("short", False, ["too_short", "no_uppercase", "no_digit", "no_special"]),
        ("toolongpasswordthatexceedslimit" * 3, False, ["too_long"]),
        ("lowercase123!", False, ["no_uppercase"]),
        ("UPPERCASE123!", False, ["no_lowercase"]),
        ("NoDigitsHere!", False, ["no_digit"]),
        ("NoSpecial123", False, ["no_special"]),
    ])
    def test_password_validation(self, password, expected_valid, expected_errors):
        """Test password validation with various strength requirements."""
        result = validate_password(password)
        
        assert result.is_valid == expected_valid
        assert set(result.errors) == set(expected_errors)

class TestPhoneValidation:
    """Test phone number validation with international formats."""
    
    @pytest.mark.parametrize("country,phone,expected", [
        ("US", "+1-555-123-4567", True),
        ("US", "(555) 123-4567", True),
        ("UK", "+44 20 7946 0958", True),
        ("NO", "+47 123 45 678", True),
        ("US", "555-123-4567", True),  # Local format
        ("US", "123", False),  # Too short
        ("US", "+1-555-123-456789", False),  # Too long
        ("INVALID", "+1-555-123-4567", False),  # Invalid country
    ])
    def test_phone_validation_by_country(self, country, phone, expected):
        """Test phone validation for different countries."""
        result = validate_phone(phone, country)
        assert result == expected
```

Parameterized tests make it easy to test edge cases and ensure your validation logic handles all scenarios correctly.

### Property-Based Testing with Hypothesis

Property-based testing generates test cases automatically, often finding edge cases that manual testing misses:

```python
# test_math_properties.py - Property-based testing
import pytest
from hypothesis import given, strategies as st, assume, example
from decimal import Decimal
from math_utils import fibonacci, prime_factors, gcd, is_prime

class TestMathematicalProperties:
    """Property-based tests for mathematical functions."""
    
    @given(n=st.integers(min_value=0, max_value=30))
    def test_fibonacci_properties(self, n):
        """Test mathematical properties of Fibonacci sequence."""
        if n <= 1:
            assert fibonacci(n) == n
        else:
            # Fibonacci property: F(n) = F(n-1) + F(n-2)
            assert fibonacci(n) == fibonacci(n-1) + fibonacci(n-2)
    
    @given(
        a=st.integers(min_value=1, max_value=1000),
        b=st.integers(min_value=1, max_value=1000)
    )
    def test_gcd_properties(self, a, b):
        """Test properties of greatest common divisor."""
        result = gcd(a, b)
        
        # GCD should divide both numbers
        assert a % result == 0
        assert b % result == 0
        
        # GCD should be symmetric
        assert gcd(a, b) == gcd(b, a)
        
        # GCD with 1 should be 1
        assert gcd(result, 1) == 1
    
    @given(n=st.integers(min_value=2, max_value=1000))
    def test_prime_factorization(self, n):
        """Test that prime factorization actually works."""
        factors = prime_factors(n)
        
        # Product of factors should equal original number
        product = 1
        for factor in factors:
            product *= factor
        assert product == n
        
        # All factors should be prime
        for factor in factors:
            assert is_prime(factor)
    
    @given(
        numbers=st.lists(
            st.decimals(min_value=Decimal('0.01'), max_value=Decimal('1000.00')),
            min_size=1,
            max_size=100
        )
    )
    def test_statistics_properties(self, numbers):
        """Test statistical calculation properties."""
        from statistics_calc import calculate_statistics
        
        stats = calculate_statistics(numbers)
        
        # Mean should be between min and max
        min_val = min(numbers)
        max_val = max(numbers)
        assert min_val <= stats.mean <= max_val
        
        # Standard deviation should be non-negative
        assert stats.std_dev >= 0
        
        # For single value, std dev should be 0
        if len(numbers) == 1:
            assert abs(stats.std_dev) < 1e-10
```

Property-based testing helps ensure your functions behave correctly across a wide range of inputs and can discover corner cases you haven't thought of.

## 3. Mocking and Test Isolation

### Isolating Units Under Test

Mocking allows you to test units in isolation by replacing dependencies with controlled test doubles:

```python
# test_email_service.py - Mocking external dependencies
import pytest
from unittest.mock import Mock, patch, MagicMock
from email_service import EmailService, EmailProvider
from user_service import UserService

class TestEmailService:
    """Test email service with mocked dependencies."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock email provider."""
        provider = Mock(spec=EmailProvider)
        provider.send_email.return_value = {"status": "sent", "message_id": "12345"}
        return provider
    
    @pytest.fixture
    def email_service(self, mock_provider):
        """Create EmailService with mocked provider."""
        return EmailService(provider=mock_provider)
    
    def test_send_welcome_email(self, email_service, mock_provider):
        """Test sending welcome email."""
        user = {"name": "Alice", "email": "alice@example.com"}
        
        result = email_service.send_welcome_email(user)
        
        # Verify the email was sent
        mock_provider.send_email.assert_called_once()
        call_args = mock_provider.send_email.call_args
        
        assert call_args[1]["to"] == "alice@example.com"
        assert "Welcome" in call_args[1]["subject"]
        assert "Alice" in call_args[1]["body"]
        assert result["status"] == "sent"
    
    def test_send_password_reset(self, email_service, mock_provider):
        """Test sending password reset email."""
        user = {"email": "user@example.com"}
        reset_token = "secure-reset-token-123"
        
        email_service.send_password_reset(user, reset_token)
        
        mock_provider.send_email.assert_called_once()
        call_args = mock_provider.send_email.call_args
        
        assert reset_token in call_args[1]["body"]
        assert "password reset" in call_args[1]["subject"].lower()
    
    @patch('email_service.generate_token')
    def test_send_email_verification(self, mock_generate_token, email_service, mock_provider):
        """Test email verification with patched token generation."""
        mock_generate_token.return_value = "verification-token-xyz"
        user = {"email": "verify@example.com"}
        
        result = email_service.send_verification_email(user)
        
        mock_generate_token.assert_called_once()
        mock_provider.send_email.assert_called_once()
        
        # Check that the generated token is used
        call_args = mock_provider.send_email.call_args
        assert "verification-token-xyz" in call_args[1]["body"]
        assert result["verification_token"] == "verification-token-xyz"

class TestEmailServiceFailures:
    """Test email service error handling."""
    
    def test_provider_failure_handling(self):
        """Test handling of provider failures."""
        provider = Mock(spec=EmailProvider)
        provider.send_email.side_effect = Exception("SMTP server unavailable")
        
        service = EmailService(provider=provider)
        user = {"email": "test@example.com"}
        
        with pytest.raises(EmailService.EmailDeliveryError) as exc_info:
            service.send_welcome_email(user)
        
        assert "SMTP server unavailable" in str(exc_info.value)
    
    @patch('email_service.time.sleep')  # Speed up retry tests
    def test_retry_mechanism(self, mock_sleep):
        """Test email retry mechanism."""
        provider = Mock(spec=EmailProvider)
        # Fail twice, then succeed
        provider.send_email.side_effect = [
            Exception("Temporary failure"),
            Exception("Still failing"),
            {"status": "sent", "message_id": "67890"}
        ]
        
        service = EmailService(provider=provider, max_retries=3)
        user = {"email": "test@example.com"}
        
        result = service.send_welcome_email(user)
        
        # Should have tried 3 times
        assert provider.send_email.call_count == 3
        assert result["status"] == "sent"
```

Mocking enables focused testing by controlling external dependencies and testing error scenarios that would be difficult to reproduce otherwise.

## 4. Integration and End-to-End Testing

### Testing Component Interactions

Integration tests verify that different parts of your system work together correctly:

```python
# test_integration.py - Integration testing examples
import pytest
from datetime import datetime, timedelta
from database import Database
from user_service import UserService
from email_service import EmailService
from notification_service import NotificationService
from test_email_provider import TestEmailProvider

@pytest.fixture(scope="session")
def test_database():
    """Session-scoped test database."""
    db = Database(":memory:")
    db.create_tables()
    yield db
    db.close()

@pytest.fixture
def integration_services(test_database):
    """Create integrated service stack for testing."""
    email_provider = TestEmailProvider()  # Test implementation
    
    services = {
        'database': test_database,
        'user_service': UserService(test_database),
        'email_service': EmailService(email_provider),
        'notification_service': NotificationService(test_database, email_provider)
    }
    
    yield services
    
    # Cleanup between tests
    test_database.clear_all_tables()

class TestUserRegistrationFlow:
    """Integration tests for complete user registration."""
    
    def test_complete_user_registration(self, integration_services):
        """Test the complete user registration flow."""
        user_service = integration_services['user_service']
        email_service = integration_services['email_service']
        notification_service = integration_services['notification_service']
        
        # Register new user
        user_data = {
            "name": "Integration Test User",
            "email": "integration@example.com",
            "password": "SecurePass123!"
        }
        
        user = user_service.register_user(**user_data)
        
        # Verify user was created
        assert user.id is not None
        assert user.email == user_data["email"]
        assert not user.is_verified
        
        # Verify verification email was sent
        sent_emails = email_service.get_sent_emails()
        assert len(sent_emails) == 1
        
        verification_email = sent_emails[0]
        assert verification_email["to"] == user.email
        assert "verify" in verification_email["subject"].lower()
        
        # Extract verification token from email
        verification_token = self._extract_token_from_email(verification_email["body"])
        
        # Verify the user's email
        result = user_service.verify_email(verification_token)
        assert result.success
        
        # Check user is now verified
        verified_user = user_service.get_user_by_id(user.id)
        assert verified_user.is_verified
        
        # Verify welcome notification was sent
        notifications = notification_service.get_user_notifications(user.id)
        assert len(notifications) >= 1
        assert any("welcome" in n.message.lower() for n in notifications)
    
    def test_user_login_flow(self, integration_services):
        """Test user authentication flow."""
        user_service = integration_services['user_service']
        
        # Create and verify user first
        user = user_service.register_user(
            "Login Test", "login@example.com", "TestPass123!"
        )
        user_service.verify_email(user.verification_token)
        
        # Test successful login
        auth_result = user_service.authenticate("login@example.com", "TestPass123!")
        assert auth_result.success
        assert auth_result.user.id == user.id
        assert auth_result.session_token is not None
        
        # Test failed login
        failed_auth = user_service.authenticate("login@example.com", "WrongPassword")
        assert not failed_auth.success
        assert failed_auth.session_token is None
    
    def test_password_reset_flow(self, integration_services):
        """Test complete password reset flow."""
        user_service = integration_services['user_service']
        email_service = integration_services['email_service']
        
        # Setup user
        user = user_service.register_user(
            "Reset Test", "reset@example.com", "OriginalPass123!"
        )
        
        # Request password reset
        reset_result = user_service.request_password_reset("reset@example.com")
        assert reset_result.success
        
        # Verify reset email was sent
        sent_emails = email_service.get_sent_emails()
        reset_emails = [e for e in sent_emails if "reset" in e["subject"].lower()]
        assert len(reset_emails) == 1
        
        # Extract reset token
        reset_token = self._extract_token_from_email(reset_emails[0]["body"])
        
        # Reset password
        new_password = "NewSecurePass456!"
        reset_result = user_service.reset_password(reset_token, new_password)
        assert reset_result.success
        
        # Verify old password no longer works
        old_auth = user_service.authenticate("reset@example.com", "OriginalPass123!")
        assert not old_auth.success
        
        # Verify new password works
        new_auth = user_service.authenticate("reset@example.com", new_password)
        assert new_auth.success
    
    def _extract_token_from_email(self, email_body: str) -> str:
        """Extract verification/reset token from email body."""
        import re
        token_pattern = r'token=([a-zA-Z0-9-_]+)'
        match = re.search(token_pattern, email_body)
        assert match, "No token found in email body"
        return match.group(1)
```

Integration tests ensure that your components work together correctly and catch issues that unit tests might miss.

## 5. Test Organization and Best Practices

### Structuring Test Suites for Maintainability

Good test organization makes tests easier to run, maintain, and understand:

```python
# conftest.py - Shared test configuration
"""Shared pytest configuration and fixtures."""

import pytest
import os
from datetime import datetime
from typing import Generator
from database import Database
from test_helpers import create_test_user, cleanup_test_files

# Configure test environment
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = ":memory:"

@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        "database_url": ":memory:",
        "email_backend": "test",
        "cache_backend": "memory",
        "debug": True,
        "testing": True
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup and cleanup for each test."""
    # Setup
    test_start_time = datetime.now()
    
    yield
    
    # Cleanup
    cleanup_test_files()
    test_duration = datetime.now() - test_start_time
    
    # Log slow tests
    if test_duration.total_seconds() > 1.0:
        print(f"Slow test detected: {test_duration.total_seconds():.2f}s")

@pytest.fixture
def temp_directory(tmp_path):
    """Create temporary directory for test files."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir

# Markers for test categorization
pytest_plugins = ["pytest_html"]  # For HTML reports

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")

# test_helpers.py - Test utility functions
"""Helper functions for testing."""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
import string

def create_test_user(
    name: str = None,
    email: str = None,
    age: int = None,
    **kwargs
) -> Dict[str, Any]:
    """Create test user data with reasonable defaults."""
    if name is None:
        name = f"Test User {random.randint(1000, 9999)}"
    
    if email is None:
        username = ''.join(random.choices(string.ascii_lowercase, k=8))
        email = f"{username}@test.example.com"
    
    if age is None:
        age = random.randint(18, 80)
    
    return {
        "name": name,
        "email": email,
        "age": age,
        "created_at": datetime.now(),
        **kwargs
    }

def create_test_users(count: int) -> List[Dict[str, Any]]:
    """Create multiple test users."""
    return [create_test_user() for _ in range(count)]

def assert_valid_timestamp(timestamp: datetime, tolerance_seconds: int = 5) -> None:
    """Assert that timestamp is recent and valid."""
    now = datetime.now()
    diff = abs((now - timestamp).total_seconds())
    assert diff <= tolerance_seconds, f"Timestamp {timestamp} is not recent enough"

def assert_valid_email_format(email: str) -> None:
    """Assert that email has valid format."""
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    assert re.match(email_pattern, email), f"Invalid email format: {email}"

def cleanup_test_files() -> None:
    """Clean up any temporary files created during tests."""
    import glob
    import os
    
    test_files = glob.glob("test_*.tmp")
    for file in test_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass
```

Shared configuration and test helpers reduce duplication and make tests more maintainable.

### Test Performance and Continuous Integration

```python
# test_performance.py - Performance testing examples
import pytest
import time
from performance_monitor import measure_time, memory_usage

class TestPerformance:
    """Performance tests to ensure code meets speed requirements."""
    
    @pytest.mark.slow
    def test_bulk_user_creation_performance(self, user_service):
        """Test that bulk operations complete within time limits."""
        user_count = 1000
        
        with measure_time() as timer:
            users = []
            for i in range(user_count):
                user = user_service.create_user(
                    f"User {i}",
                    f"user{i}@example.com",
                    25
                )
                users.append(user)
        
        # Should create 1000 users in under 2 seconds
        assert timer.elapsed < 2.0, f"Bulk creation took {timer.elapsed:.2f}s"
        assert len(users) == user_count
    
    @pytest.mark.slow
    def test_search_performance(self, populated_user_service):
        """Test search performance with large dataset."""
        # populated_user_service fixture creates 10,000 users
        
        with measure_time() as timer:
            results = populated_user_service.search_users("Smith")
        
        # Search should complete in under 100ms
        assert timer.elapsed < 0.1, f"Search took {timer.elapsed:.3f}s"
        assert len(results) > 0
    
    def test_memory_usage(self, user_service):
        """Test that operations don't cause memory leaks."""
        initial_memory = memory_usage()
        
        # Perform operations that should not leak memory
        for i in range(100):
            user = user_service.create_user(f"User {i}", f"user{i}@test.com", 25)
            user_service.delete_user(user.id)
        
        final_memory = memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 10MB)
        assert memory_increase < 10_000_000, f"Memory increased by {memory_increase} bytes"

# pytest.ini - Project test configuration
"""
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --html=reports/report.html
    --self-contained-html
    --cov=src
    --cov-report=html:reports/coverage
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    smoke: marks tests as smoke tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
"""
```

Performance tests and proper CI configuration ensure your code meets quality standards in production environments.

## Key Takeaways

**Start with Unit Tests**: Focus on testing individual functions and methods in isolation. Unit tests are fast, focused, and form the foundation of your test suite.

**Use Fixtures Effectively**: Leverage pytest fixtures to manage test data and dependencies. Fixtures promote test isolation and make complex setups reusable.

**Practice Property-Based Testing**: Use Hypothesis to automatically generate test cases and find edge cases you might miss with manual testing.

**Mock External Dependencies**: Use mocking to isolate units under test and control external dependencies. This makes tests faster, more reliable, and easier to debug.

**Write Integration Tests**: Test how your components work together. Integration tests catch issues that unit tests might miss and verify your system works end-to-end.

**Organize Tests Well**: Use clear naming conventions, shared fixtures, and proper test categorization. Good organization makes tests easier to maintain and run.

**Test Edge Cases**: Don't just test the happy path. Test error conditions, boundary values, and unexpected inputs to build robust software.

**Measure Test Coverage**: Use coverage tools to identify untested code, but remember that 100% coverage doesn't guarantee bug-free code.

**Keep Tests Fast**: Most of your tests should run quickly. Use mocking and in-memory databases to keep test suites fast and encourage frequent running.

**Test Behavior, Not Implementation**: Focus on testing what your code does, not how it does it. This makes tests more resilient to refactoring.

Remember: Good tests are an investment in your codebase's future. They catch bugs early, enable confident refactoring, and serve as living documentation of how your system should behave. The time spent writing tests pays dividends in reduced debugging time and increased confidence when making changes.
