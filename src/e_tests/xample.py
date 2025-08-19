"""
Testing Workshop - SOLUTIONS

Complete solutions for all testing tasks demonstrating comprehensive testing practices.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest
from hypothesis import given, strategies as st, assume

# =============================================================================
# TASK 1: Basic Unit Testing with pytest - SOLUTION
# =============================================================================

class Calculator:
    """Calculator with basic operations and memory functionality."""
    
    def __init__(self):
        self._memory: float | None = None
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers."""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return base ** exponent
    
    def store(self, value: float) -> None:
        """Store value in memory."""
        self._memory = value
    
    def recall(self) -> float | None:
        """Recall value from memory."""
        return self._memory
    
    def clear_memory(self) -> None:
        """Clear memory."""
        self._memory = None

class TestCalculator:
    """Comprehensive test suite for Calculator class."""
    
    @pytest.fixture
    def calc(self):
        """Create a fresh calculator for each test."""
        return Calculator()
    
    def test_addition(self, calc):
        """Test addition operation."""
        assert calc.add(2, 3) == 5
        assert calc.add(-1, 1) == 0
        assert calc.add(0.1, 0.2) == pytest.approx(0.3)
    
    def test_subtraction(self, calc):
        """Test subtraction operation."""
        assert calc.subtract(5, 3) == 2
        assert calc.subtract(1, 1) == 0
        assert calc.subtract(-1, -1) == 0
    
    def test_multiplication(self, calc):
        """Test multiplication operation."""
        assert calc.multiply(3, 4) == 12
        assert calc.multiply(-2, 3) == -6
        assert calc.multiply(0, 100) == 0
    
    def test_division(self, calc):
        """Test division operation."""
        assert calc.divide(6, 2) == 3
        assert calc.divide(1, 3) == pytest.approx(0.3333333)
        assert calc.divide(-6, 2) == -3
    
    def test_division_by_zero(self, calc):
        """Test division by zero raises error."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(5, 0)
    
    def test_power(self, calc):
        """Test power operation."""
        assert calc.power(2, 3) == 8
        assert calc.power(4, 0.5) == 2
        assert calc.power(5, 0) == 1
    
    def test_memory_operations(self, calc):
        """Test memory functionality."""
        # Initially no memory
        assert calc.recall() is None
        
        # Store and recall
        calc.store(42.5)
        assert calc.recall() == 42.5
        
        # Memory persists
        assert calc.recall() == 42.5
        
        # Clear memory
        calc.clear_memory()
        assert calc.recall() is None

# =============================================================================
# TASK 2: Fixtures and Test Data Management - SOLUTION
# =============================================================================

class User:
    """User data class."""
    
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
        self.created_at = datetime.now()
        self.is_active = True

class UserManager:
    """User management system."""
    
    def __init__(self):
        self._users: dict[int, User] = {}
        self._next_id = 1
    
    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(self._next_id, name, email)
        self._users[user.id] = user
        self._next_id += 1
        return user
    
    def get_user(self, user_id: int) -> User | None:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def update_user(self, user_id: int, **kwargs) -> bool:
        """Update user attributes."""
        user = self.get_user(user_id)
        if not user:
            return False
        
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        return True
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user by ID."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False
    
    def list_users(self, active_only: bool = True) -> list[User]:
        """list all users."""
        users = list(self._users.values())
        if active_only:
            users = [u for u in users if u.is_active]
        return users

@pytest.fixture
def user_manager():
    """Create a fresh UserManager for each test."""
    return UserManager()

@pytest.fixture
def sample_users():
    """Sample user data for testing."""
    return [
        {"name": "Alice Smith", "email": "alice@example.com"},
        {"name": "Bob Jones", "email": "bob@example.com"},
        {"name": "Charlie Brown", "email": "charlie@example.com"}
    ]

@pytest.fixture
def populated_manager(user_manager, sample_users):
    """UserManager populated with sample users."""
    users = []
    for user_data in sample_users:
        user = user_manager.create_user(**user_data)
        users.append(user)
    return user_manager, users

class TestUserManager:
    """Test suite using fixtures."""
    
    def test_create_user(self, user_manager):
        """Test user creation."""
        user = user_manager.create_user("Test User", "test@example.com")
        
        assert user.id == 1
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)
    
    def test_get_user(self, populated_manager):
        """Test getting user."""
        manager, users = populated_manager
        
        user = manager.get_user(users[0].id)
        assert user is not None
        assert user.name == "Alice Smith"
        
        # Test non-existent user
        assert manager.get_user(999) is None
    
    def test_update_user(self, populated_manager):
        """Test updating user."""
        manager, users = populated_manager
        
        success = manager.update_user(users[0].id, name="Alice Johnson")
        assert success is True
        
        user = manager.get_user(users[0].id)
        assert user.name == "Alice Johnson"
    
    def test_delete_user(self, populated_manager):
        """Test deleting user."""
        manager, users = populated_manager
        
        success = manager.delete_user(users[0].id)
        assert success is True
        
        assert manager.get_user(users[0].id) is None
    
    def test_list_users(self, populated_manager):
        """Test listing users."""
        manager, users = populated_manager
        
        all_users = manager.list_users(active_only=False)
        assert len(all_users) == 3
        
        # Deactivate one user
        manager.update_user(users[0].id, is_active=False)
        active_users = manager.list_users(active_only=True)
        assert len(active_users) == 2

# =============================================================================
# TASK 3: Parameterized Testing - SOLUTION
# =============================================================================

class ValidationResult:
    """Result of password validation."""
    
    def __init__(self, valid: bool, errors: list[str] = None):
        self.valid = valid
        self.errors = errors or []

def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> ValidationResult:
    """Validate password strength."""
    errors = []
    
    if len(password) < 8:
        errors.append("too_short")
    
    if not any(c.isupper() for c in password):
        errors.append("no_uppercase")
    
    if not any(c.islower() for c in password):
        errors.append("no_lowercase")
    
    if not any(c.isdigit() for c in password):
        errors.append("no_digit")
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        errors.append("no_special")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)

def validate_age(age: int) -> bool:
    """Validate age range."""
    return 0 <= age <= 150

class TestValidation:
    """Parameterized test suite for validation functions."""
    
    @pytest.mark.parametrize("email,expected", [
        ("user@example.com", True),
        ("test.user+tag@domain.co.uk", True),
        ("simple@domain.org", True),
        ("user123@test-domain.com", True),
        ("invalid.email", False),
        ("@domain.com", False),
        ("user@", False),
        ("", False),
        ("user@@domain.com", False),
        ("user@domain", False),
    ])
    def test_email_validation(self, email, expected):
        """Test email validation with various formats."""
        assert validate_email(email) == expected
    
    @pytest.mark.parametrize("password,expected_valid,expected_errors", [
        ("ValidPass123!", True, []),
        ("short", False, ["too_short", "no_uppercase", "no_digit", "no_special"]),
        ("onlylowercase", False, ["no_uppercase", "no_digit", "no_special"]),
        ("ONLYUPPERCASE", False, ["no_lowercase", "no_digit", "no_special"]),
        ("NoDigitsHere!", False, ["no_digit"]),
        ("NoSpecial123", False, ["no_special"]),
        ("Valid123!", True, []),
        ("Another$ecure1", True, []),
    ])
    def test_password_validation(self, password, expected_valid, expected_errors):
        """Test password validation with various strengths."""
        result = validate_password(password)
        assert result.valid == expected_valid
        assert set(result.errors) == set(expected_errors)
    
    @pytest.mark.parametrize("age,expected", [
        (0, True),
        (25, True),
        (150, True),
        (-1, False),
        (151, False),
        (1000, False),
    ])
    def test_age_validation(self, age, expected):
        """Test age validation with boundary values."""
        assert validate_age(age) == expected

# =============================================================================
# TASK 4: Mocking External Dependencies - SOLUTION
# =============================================================================

class EmailProvider:
    """External email provider interface."""
    
    def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        """Send email and return result."""
        # This would integrate with real email service
        raise NotImplementedError("Real implementation would call external service")

class EmailService:
    """Email service with retry mechanism."""
    
    def __init__(self, provider: EmailProvider, max_retries: int = 3):
        self.provider = provider
        self.max_retries = max_retries
    
    def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        """Send email with retry mechanism."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self.provider.send_email(to, subject, body)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        raise Exception(f"Failed to send email after {self.max_retries} attempts: {last_error}")
    
    def send_welcome_email(self, user: dict[str, str]) -> dict[str, Any]:
        """Send welcome email to user."""
        subject = f"Welcome, {user['name']}!"
        body = f"Hello {user['name']},\n\nWelcome to our service!"
        return self.send_email(user['email'], subject, body)
    
    def send_password_reset(self, user: dict[str, str], reset_token: str) -> dict[str, Any]:
        """Send password reset email."""
        subject = "Password Reset Request"
        body = f"Click here to reset your password: https://example.com/reset?token={reset_token}"
        return self.send_email(user['email'], subject, body)

class TestEmailService:
    """Test suite with mocking."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create mock email provider."""
        provider = Mock(spec=EmailProvider)
        provider.send_email.return_value = {"status": "sent", "message_id": "12345"}
        return provider
    
    @pytest.fixture
    def email_service(self, mock_provider):
        """Create EmailService with mock provider."""
        return EmailService(mock_provider)
    
    def test_send_email_success(self, email_service, mock_provider):
        """Test successful email sending."""
        result = email_service.send_email("test@example.com", "Subject", "Body")
        
        mock_provider.send_email.assert_called_once_with("test@example.com", "Subject", "Body")
        assert result["status"] == "sent"
    
    def test_send_welcome_email(self, email_service, mock_provider):
        """Test welcome email sending."""
        user = {"name": "Alice", "email": "alice@example.com"}
        
        result = email_service.send_welcome_email(user)
        
        mock_provider.send_email.assert_called_once()
        call_args = mock_provider.send_email.call_args
        
        assert call_args[0][0] == "alice@example.com"
        assert "Welcome, Alice!" in call_args[0][1]
        assert "Alice" in call_args[0][2]
    
    def test_send_password_reset(self, email_service, mock_provider):
        """Test password reset email."""
        user = {"email": "user@example.com"}
        reset_token = "secure-token-123"
        
        email_service.send_password_reset(user, reset_token)
        
        call_args = mock_provider.send_email.call_args
        assert reset_token in call_args[0][2]
        assert "Password Reset" in call_args[0][1]
    
    def test_retry_mechanism(self, mock_provider):
        """Test retry mechanism on failure."""
        # Fail twice, then succeed
        mock_provider.send_email.side_effect = [
            Exception("Network error"),
            Exception("Still failing"),
            {"status": "sent", "message_id": "67890"}
        ]
        
        service = EmailService(mock_provider, max_retries=3)
        result = service.send_email("test@example.com", "Subject", "Body")
        
        assert mock_provider.send_email.call_count == 3
        assert result["status"] == "sent"
    
    def test_max_retries_exceeded(self, mock_provider):
        """Test failure after max retries."""
        mock_provider.send_email.side_effect = Exception("Persistent error")
        
        service = EmailService(mock_provider, max_retries=2)
        
        with pytest.raises(Exception, match="Failed to send email after 2 attempts"):
            service.send_email("test@example.com", "Subject", "Body")

# =============================================================================
# TASK 5: Property-Based Testing with Hypothesis - SOLUTION
# =============================================================================

def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(n: int) -> bool:
    """Check if number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

class TestMathematicalProperties:
    """Property-based tests for mathematical functions."""
    
    @given(n=st.integers(min_value=0, max_value=20))
    def test_factorial_properties(self, n):
        """Test factorial properties."""
        result = factorial(n)
        
        # Factorial should be positive
        assert result > 0
        
        # Factorial should be increasing
        if n > 0:
            assert factorial(n) > factorial(n - 1)
        
        # 0! = 1! = 1
        if n in (0, 1):
            assert result == 1
    
    @given(n=st.integers(min_value=0, max_value=30))
    def test_fibonacci_properties(self, n):
        """Test Fibonacci properties."""
        if n <= 1:
            assert fibonacci(n) == n
        else:
            # Fibonacci recurrence relation
            assert fibonacci(n) == fibonacci(n-1) + fibonacci(n-2)
    
    @given(
        a=st.integers(min_value=1, max_value=1000),
        b=st.integers(min_value=1, max_value=1000)
    )
    def test_gcd_properties(self, a, b):
        """Test GCD properties."""
        result = gcd(a, b)
        
        # GCD should divide both numbers
        assert a % result == 0
        assert b % result == 0
        
        # GCD should be symmetric
        assert gcd(a, b) == gcd(b, a)
        
        # GCD should be associative for GCD(a, 1) = 1
        if result > 1:
            assert gcd(result, 1) == 1
    
    @given(n=st.integers(min_value=2, max_value=100))
    def test_prime_properties(self, n):
        """Test prime number properties."""
        result = is_prime(n)
        
        if result:  # If n is prime
            # Should only be divisible by 1 and itself
            divisors = [i for i in range(2, n) if n % i == 0]
            assert len(divisors) == 0
        else:  # If n is not prime
            # Should have at least one divisor other than 1 and itself
            if n > 1:
                divisors = [i for i in range(2, n) if n % i == 0]
                assert len(divisors) > 0

# =============================================================================
# TASK 6: Integration Testing - SOLUTION
# =============================================================================

class Author:
    """Author data class."""
    
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
        self.posts: list[int] = []  # Post IDs

class Post:
    """Post data class."""
    
    def __init__(self, id: int, title: str, content: str, author_id: int):
        self.id = id
        self.title = title
        self.content = content
        self.author_id = author_id
        self.created_at = datetime.now()
        self.published = False

class Comment:
    """Comment data class."""
    
    def __init__(self, id: int, post_id: int, author_name: str, content: str):
        self.id = id
        self.post_id = post_id
        self.author_name = author_name
        self.content = content
        self.created_at = datetime.now()

class Blog:
    """Blog management system."""
    
    def __init__(self):
        self._authors: dict[int, Author] = {}
        self._posts: dict[int, Post] = {}
        self._next_author_id = 1
        self._next_post_id = 1
    
    def create_author(self, name: str, email: str) -> Author:
        """Create a new author."""
        author = Author(self._next_author_id, name, email)
        self._authors[author.id] = author
        self._next_author_id += 1
        return author
    
    def create_post(self, author_id: int, title: str, content: str) -> Post:
        """Create a new post."""
        if author_id not in self._authors:
            raise ValueError("Author not found")
        
        post = Post(self._next_post_id, title, content, author_id)
        self._posts[post.id] = post
        self._authors[author_id].posts.append(post.id)
        self._next_post_id += 1
        return post
    
    def publish_post(self, post_id: int) -> bool:
        """Publish a post."""
        post = self._posts.get(post_id)
        if post:
            post.published = True
            return True
        return False
    
    def get_published_posts(self) -> list[Post]:
        """Get all published posts."""
        return [post for post in self._posts.values() if post.published]

class CommentService:
    """Comment management service."""
    
    def __init__(self):
        self._comments: dict[int, Comment] = {}
        self._next_comment_id = 1
    
    def add_comment(self, post_id: int, author_name: str, content: str) -> Comment:
        """Add a comment to a post."""
        comment = Comment(self._next_comment_id, post_id, author_name, content)
        self._comments[comment.id] = comment
        self._next_comment_id += 1
        return comment
    
    def get_post_comments(self, post_id: int) -> list[Comment]:
        """Get all comments for a post."""
        return [c for c in self._comments.values() if c.post_id == post_id]

class TestBlogIntegration:
    """Integration tests for blog system."""
    
    @pytest.fixture
    def blog_system(self):
        """Create blog system with comment service."""
        blog = Blog()
        comment_service = CommentService()
        return blog, comment_service
    
    def test_complete_blog_workflow(self, blog_system):
        """Test complete blog workflow."""
        blog, comment_service = blog_system
        
        # Create author
        author = blog.create_author("Alice Writer", "alice@example.com")
        assert author.id == 1
        assert author.name == "Alice Writer"
        
        # Create post
        post = blog.create_post(author.id, "My First Post", "This is great content!")
        assert post.id == 1
        assert post.author_id == author.id
        assert not post.published
        
        # Publish post
        success = blog.publish_post(post.id)
        assert success is True
        
        published_posts = blog.get_published_posts()
        assert len(published_posts) == 1
        assert published_posts[0].id == post.id
        
        # Add comments
        comment1 = comment_service.add_comment(post.id, "Reader1", "Great post!")
        comment2 = comment_service.add_comment(post.id, "Reader2", "Very informative")
        
        comments = comment_service.get_post_comments(post.id)
        assert len(comments) == 2
        assert comments[0].content == "Great post!"
        assert comments[1].content == "Very informative"
    
    def test_multiple_authors_and_posts(self, blog_system):
        """Test multiple authors creating posts."""
        blog, comment_service = blog_system
        
        # Create multiple authors
        author1 = blog.create_author("Author 1", "author1@example.com")
        author2 = blog.create_author("Author 2", "author2@example.com")
        
        # Each author creates posts
        post1 = blog.create_post(author1.id, "Post 1", "Content 1")
        post2 = blog.create_post(author2.id, "Post 2", "Content 2")
        post3 = blog.create_post(author1.id, "Post 3", "Content 3")
        
        # Publish some posts
        blog.publish_post(post1.id)
        blog.publish_post(post3.id)
        
        published_posts = blog.get_published_posts()
        assert len(published_posts) == 2
        
        # Check author associations
        assert author1.posts == [post1.id, post3.id]
        assert author2.posts == [post2.id]

# =============================================================================
# TASK 7: Exception Testing and Error Handling - SOLUTION
# =============================================================================

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error

class BackupError(Exception):
    """Custom exception for backup errors."""
    pass

class FileProcessor:
    """File processor with comprehensive error handling."""
    
    def __init__(self, backup_dir: str = None):
        self.backup_dir = backup_dir or tempfile.gettempdir()
    
    def read_json(self, file_path: str) -> dict[str, Any]:
        """Read JSON file with error handling."""
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise ProcessingError(f"File not found: {file_path}") from e
        except PermissionError as e:
            raise ProcessingError(f"Permission denied: {file_path}") from e
        except json.JSONDecodeError as e:
            raise ProcessingError(f"Invalid JSON in {file_path}: {str(e)}", e) from e
    
    def write_json(self, file_path: str, data: dict[str, Any]) -> None:
        """Write JSON file with error handling."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except PermissionError as e:
            raise ProcessingError(f"Permission denied writing to: {file_path}") from e
        except OSError as e:
            raise ProcessingError(f"OS error writing to {file_path}: {str(e)}", e) from e
    
    def backup_file(self, file_path: str) -> str:
        """Backup file to backup directory."""
        if not os.path.exists(file_path):
            raise BackupError(f"Cannot backup non-existent file: {file_path}")
        
        try:
            backup_name = f"{os.path.basename(file_path)}.backup.{int(time.time())}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            shutil.copy2(file_path, backup_path)
            return backup_path
        except PermissionError as err:
            raise BackupError(f"Permission denied creating backup: {backup_path}") from err
        except OSError as e:
            raise BackupError(f"Failed to create backup: {str(e)}") from e
    
    def process_batch(self, file_paths: list[str]) -> dict[str, Any]:
        """Process multiple files, collecting errors."""
        results = {"processed": [], "errors": []}
        
        for file_path in file_paths:
            try:
                data = self.read_json(file_path)
                # Simulate processing
                processed_data = {"file": file_path, "records": len(data), "status": "success"}
                results["processed"].append(processed_data)
            except ProcessingError as e:
                results["errors"].append({"file": file_path, "error": str(e)})
        
        return results

class TestFileProcessor:
    """Exception testing suite for FileProcessor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def processor(self, temp_dir):
        """Create FileProcessor with temp directory."""
        return FileProcessor(backup_dir=temp_dir)
    
    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create sample JSON file for testing."""
        file_path = os.path.join(temp_dir, "sample.json")
        data = {"name": "test", "value": 42, "items": [1, 2, 3]}
        with open(file_path, 'w') as f:
            json.dump(data, f)
        return file_path
    
    def test_read_json_success(self, processor, sample_json_file):
        """Test successful JSON reading."""
        data = processor.read_json(sample_json_file)
        assert data["name"] == "test"
        assert data["value"] == 42
        assert data["items"] == [1, 2, 3]
    
    def test_read_json_file_not_found(self, processor):
        """Test FileNotFoundError handling."""
        with pytest.raises(ProcessingError, match="File not found"):
            processor.read_json("nonexistent.json")
    
    def test_read_json_invalid_json(self, processor, temp_dir):
        """Test JSON decode error handling."""
        invalid_file = os.path.join(temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(ProcessingError, match="Invalid JSON"):
            processor.read_json(invalid_file)
    
    def test_write_json_success(self, processor, temp_dir):
        """Test successful JSON writing."""
        file_path = os.path.join(temp_dir, "output.json")
        data = {"test": "data"}
        
        processor.write_json(file_path, data)
        
        # Verify file was written correctly
        with open(file_path) as f:
            written_data = json.load(f)
        assert written_data == data
    
    def test_backup_file_success(self, processor, sample_json_file):
        """Test successful file backup."""
        backup_path = processor.backup_file(sample_json_file)
        
        assert os.path.exists(backup_path)
        assert backup_path.endswith(".backup." + str(int(time.time())))
        
        # Verify backup content matches original
        with open(sample_json_file) as f:
            original_data = json.load(f)
        with open(backup_path) as f:
            backup_data = json.load(f)
        assert original_data == backup_data
    
    def test_backup_file_not_found(self, processor):
        """Test backup of non-existent file."""
        with pytest.raises(BackupError, match="Cannot backup non-existent file"):
            processor.backup_file("nonexistent.json")
    
    def test_process_batch_mixed_results(self, processor, temp_dir):
        """Test batch processing with mixed success/failure."""
        # Create valid file
        valid_file = os.path.join(temp_dir, "valid.json")
        with open(valid_file, 'w') as f:
            json.dump({"data": "valid"}, f)
        
        # Create invalid file
        invalid_file = os.path.join(temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("invalid json")
        
        files = [valid_file, invalid_file, "nonexistent.json"]
        results = processor.process_batch(files)
        
        assert len(results["processed"]) == 1
        assert len(results["errors"]) == 2
        assert results["processed"][0]["file"] == valid_file
        assert results["processed"][0]["status"] == "success"

# =============================================================================
# TASK 8: Test Performance and Optimization - SOLUTION
# =============================================================================

class DataAnalyzer:
    """Data analysis tool with performance optimization."""
    
    def __init__(self):
        self._data: list[float] = []
        self._stats_cache: dict[str, float] | None = None
    
    def load_data(self, data: list[float]) -> None:
        """Load data for analysis."""
        self._data = data.copy()
        self._stats_cache = None  # Invalidate cache
    
    def calculate_statistics(self) -> dict[str, float]:
        """Calculate statistical measures."""
        if not self._data:
            raise ValueError("No data loaded")
        
        if self._stats_cache is not None:
            return self._stats_cache
        
        # Calculate statistics
        sorted_data = sorted(self._data)
        n = len(sorted_data)
        
        stats = {
            "mean": statistics.mean(self._data),
            "median": statistics.median(self._data),
            "std_dev": statistics.stdev(self._data) if n > 1 else 0.0,
            "min": min(self._data),
            "max": max(self._data),
            "count": n
        }
        
        self._stats_cache = stats
        return stats
    
    def find_outliers(self, threshold: float = 2.0) -> list[float]:
        """Find outliers beyond threshold standard deviations."""
        if not self._data:
            return []
        
        stats = self.calculate_statistics()
        mean = stats["mean"]
        std_dev = stats["std_dev"]
        
        if std_dev == 0:
            return []
        
        outliers = []
        for value in self._data:
            z_score = abs((value - mean) / std_dev)
            if z_score > threshold:
                outliers.append(value)
        
        return outliers
    
    def generate_report(self) -> str:
        """Generate analysis report."""
        if not self._data:
            return "No data available for analysis."
        
        stats = self.calculate_statistics()
        outliers = self.find_outliers()
        
        report = f"""
Data Analysis Report
====================
Sample Size: {stats['count']}
Mean: {stats['mean']:.2f}
Median: {stats['median']:.2f}
Standard Deviation: {stats['std_dev']:.2f}
Range: {stats['min']:.2f} - {stats['max']:.2f}
Outliers Found: {len(outliers)}
        """.strip()
        
        return report

class TestDataAnalyzerPerformance:
    """Performance test suite for DataAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create DataAnalyzer instance."""
        return DataAnalyzer()
    
    @pytest.fixture
    def small_dataset(self):
        """Small dataset for basic testing."""
        return [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # 100.0 is outlier
    
    @pytest.fixture
    def large_dataset(self):
        """Large dataset for performance testing."""
        import random
        random.seed(42)  # Reproducible results
        return [random.gauss(50, 10) for _ in range(10000)]
    
    def test_basic_statistics(self, analyzer, small_dataset):
        """Test basic statistical calculations."""
        analyzer.load_data(small_dataset)
        stats = analyzer.calculate_statistics()
        
        assert stats["count"] == 6
        assert stats["min"] == 1.0
        assert stats["max"] == 100.0
        assert stats["median"] == 3.5  # Average of 3.0 and 4.0
    
    def test_outlier_detection(self, analyzer, small_dataset):
        """Test outlier detection."""
        analyzer.load_data(small_dataset)
        outliers = analyzer.find_outliers(threshold=2.0)
        
        assert 100.0 in outliers
        assert len(outliers) >= 1
    
    def test_caching_performance(self, analyzer, large_dataset):
        """Test that statistics caching improves performance."""
        analyzer.load_data(large_dataset)
        
        # First calculation (no cache)
        start_time = time.time()
        stats1 = analyzer.calculate_statistics()
        first_duration = time.time() - start_time
        
        # Second calculation (with cache)
        start_time = time.time()
        stats2 = analyzer.calculate_statistics()
        second_duration = time.time() - start_time
        
        assert stats1 == stats2  # Results should be identical
        assert second_duration < first_duration * 0.1  # Cache should be much faster
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, analyzer, large_dataset):
        """Test performance with large dataset."""
        start_time = time.time()
        analyzer.load_data(large_dataset)
        stats = analyzer.calculate_statistics()
        outliers = analyzer.find_outliers()
        report = analyzer.generate_report()
        duration = time.time() - start_time
        
        # Should complete analysis in under 1 second
        assert duration < 1.0
        assert stats["count"] == 10000
        assert isinstance(outliers, list)
        assert "Data Analysis Report" in report
    
    def test_memory_efficiency(self, analyzer):
        """Test memory usage doesn't grow with repeated operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple operations
        for i in range(10):
            data = [float(j) for j in range(1000)]
            analyzer.load_data(data)
            analyzer.calculate_statistics()
            analyzer.find_outliers()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024

# =============================================================================
# TASK 9: Test Organization and Best Practices - SOLUTION
# =============================================================================

class Product:
    """Product in e-commerce system."""
    
    def __init__(self, id: int, name: str, price: float, category: str, in_stock: bool = True):
        self.id = id
        self.name = name
        self.price = price
        self.category = category
        self.in_stock = in_stock
        self.weight = 1.0  # Default weight in kg

class Cart:
    """Shopping cart with items and discounts."""
    
    def __init__(self):
        self._items: dict[int, dict[str, Any]] = {}  # product_id -> {product, quantity}
        self._discounts: list['Discount'] = []
    
    def add_item(self, product: Product, quantity: int = 1) -> None:
        """Add item to cart."""
        if not product.in_stock:
            raise ValueError(f"Product {product.name} is out of stock")
        
        if product.id in self._items:
            self._items[product.id]["quantity"] += quantity
        else:
            self._items[product.id] = {"product": product, "quantity": quantity}
    
    def remove_item(self, product_id: int) -> bool:
        """Remove item from cart."""
        if product_id in self._items:
            del self._items[product_id]
            return True
        return False
    
    def update_quantity(self, product_id: int, quantity: int) -> bool:
        """Update item quantity."""
        if product_id in self._items:
            if quantity <= 0:
                return self.remove_item(product_id)
            self._items[product_id]["quantity"] = quantity
            return True
        return False
    
    def apply_discount(self, discount: 'Discount') -> None:
        """Apply discount to cart."""
        self._discounts.append(discount)
    
    def get_subtotal(self) -> float:
        """Calculate subtotal before discounts."""
        subtotal = 0.0
        for item in self._items.values():
            subtotal += item["product"].price * item["quantity"]
        return subtotal
    
    def get_total_weight(self) -> float:
        """Calculate total weight of items."""
        total_weight = 0.0
        for item in self._items.values():
            total_weight += item["product"].weight * item["quantity"]
        return total_weight
    
    def calculate_total(self, include_shipping: bool = False, distance: float = 0) -> float:
        """Calculate total price with discounts and optional shipping."""
        subtotal = self.get_subtotal()
        
        # Apply discounts
        total = subtotal
        for discount in self._discounts:
            total = discount.apply(total)
        
        # Add shipping if requested
        if include_shipping:
            shipping_calc = ShippingCalculator()
            shipping_cost = shipping_calc.calculate(self, distance)
            total += shipping_cost
        
        return total
    
    def get_items(self) -> list[dict[str, Any]]:
        """Get all items in cart."""
        return list(self._items.values())

class Discount:
    """Discount application for cart."""
    
    def __init__(self, code: str, percentage: float = 0, fixed_amount: float = 0):
        self.code = code
        self.percentage = percentage
        self.fixed_amount = fixed_amount
    
    def apply(self, amount: float) -> float:
        """Apply discount to amount."""
        if self.percentage > 0:
            return amount * (1 - self.percentage)
        elif self.fixed_amount > 0:
            return max(0, amount - self.fixed_amount)
        return amount

class ShippingCalculator:
    """Calculate shipping costs based on weight and distance."""
    
    def __init__(self, base_rate: float = 5.0, weight_rate: float = 2.0, distance_rate: float = 0.1):
        self.base_rate = base_rate
        self.weight_rate = weight_rate
        self.distance_rate = distance_rate
    
    def calculate(self, cart: Cart, distance: float) -> float:
        """Calculate shipping cost."""
        if distance <= 0:
            return 0.0
        
        weight = cart.get_total_weight()
        shipping_cost = self.base_rate + (weight * self.weight_rate) + (distance * self.distance_rate)
        return shipping_cost

# Test organization with proper markers and helpers
pytest.fixture
def sample_products():
    """Sample products for testing."""
    return [
        Product(1, "Laptop", 999.99, "Electronics"),
        Product(2, "Mouse", 29.99, "Electronics"),
        Product(3, "Book", 15.99, "Books"),
        Product(4, "Headphones", 199.99, "Electronics", in_stock=False)
    ]

@pytest.fixture
def empty_cart():
    """Empty shopping cart."""
    return Cart()

@pytest.fixture
def populated_cart(empty_cart, sample_products):
    """Cart with sample items."""
    cart = empty_cart
    cart.add_item(sample_products[0], 1)  # Laptop
    cart.add_item(sample_products[1], 2)  # 2 Mice
    return cart

@pytest.mark.unit
class TestProduct:
    """Unit tests for Product class."""
    
    def test_product_creation(self):
        """Test product creation."""
        product = Product(1, "Test Item", 99.99, "Test Category")
        
        assert product.id == 1
        assert product.name == "Test Item"
        assert product.price == 99.99
        assert product.category == "Test Category"
        assert product.in_stock is True
        assert product.weight == 1.0

@pytest.mark.unit
class TestCart:
    """Unit tests for Cart class."""
    
    def test_add_item(self, empty_cart, sample_products):
        """Test adding items to cart."""
        cart = empty_cart
        product = sample_products[0]
        
        cart.add_item(product, 2)
        items = cart.get_items()
        
        assert len(items) == 1
        assert items[0]["product"] == product
        assert items[0]["quantity"] == 2
    
    def test_add_out_of_stock_item(self, empty_cart, sample_products):
        """Test adding out of stock item fails."""
        cart = empty_cart
        out_of_stock_product = sample_products[3]
        
        with pytest.raises(ValueError, match="out of stock"):
            cart.add_item(out_of_stock_product)
    
    def test_remove_item(self, populated_cart):
        """Test removing items from cart."""
        cart = populated_cart
        initial_count = len(cart.get_items())
        
        success = cart.remove_item(1)  # Remove laptop
        assert success is True
        assert len(cart.get_items()) == initial_count - 1
    
    def test_calculate_subtotal(self, populated_cart):
        """Test subtotal calculation."""
        cart = populated_cart
        # Laptop (999.99) + 2 Mice (29.99 each) = 1059.97
        expected = 999.99 + (29.99 * 2)
        assert cart.get_subtotal() == pytest.approx(expected)

@pytest.mark.unit
class TestDiscount:
    """Unit tests for Discount class."""
    
    def test_percentage_discount(self):
        """Test percentage discount application."""
        discount = Discount("SAVE10", percentage=0.10)
        result = discount.apply(100.0)
        assert result == 90.0
    
    def test_fixed_amount_discount(self):
        """Test fixed amount discount application."""
        discount = Discount("SAVE20", fixed_amount=20.0)
        result = discount.apply(100.0)
        assert result == 80.0
    
    def test_fixed_amount_not_below_zero(self):
        """Test fixed discount doesn't go below zero."""
        discount = Discount("BIGDISCOUNT", fixed_amount=150.0)
        result = discount.apply(100.0)
        assert result == 0.0

@pytest.mark.integration
class TestEcommerceIntegration:
    """Integration tests for e-commerce workflow."""
    
    def test_complete_purchase_workflow(self, sample_products):
        """Test complete purchase process."""
        cart = Cart()
        
        # Add items to cart
        cart.add_item(sample_products[0], 1)  # Laptop
        cart.add_item(sample_products[2], 3)  # 3 Books
        
        # Apply discount
        discount = Discount("WELCOME10", percentage=0.10)
        cart.apply_discount(discount)
        
        # Calculate total with shipping
        total = cart.calculate_total(include_shipping=True, distance=50.0)
        
        # Verify calculations
        subtotal = 999.99 + (15.99 * 3)  # 1047.96
        discounted = subtotal * 0.9  # 943.164
        
        # Shipping: base(5) + weight(5*2) + distance(50*0.1) = 20
        expected_total = discounted + 20.0
        
        assert total == pytest.approx(expected_total, rel=1e-2)
    
    @pytest.mark.slow
    def test_large_cart_performance(self, sample_products):
        """Test performance with large number of items."""
        cart = Cart()
        
        # Add many items
        start_time = time.time()
        for i in range(1000):
            product_index = i % len(sample_products[:3])  # Only in-stock products
            cart.add_item(sample_products[product_index], 1)
        
        total = cart.calculate_total()
        duration = time.time() - start_time
        
        # Should complete in reasonable time
        assert duration < 1.0
        assert total > 0

# =============================================================================
# BONUS TASK: Advanced Testing Patterns - SOLUTION
# =============================================================================

# Custom pytest fixtures
@pytest.fixture(scope="session")
def test_database():
    """Session-scoped test database fixture."""
    # This would set up a test database
    db_path = tempfile.mktemp(suffix=".db")
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def api_client():
    """Mock API client for testing."""
    class MockAPIClient:
        def __init__(self):
            self.responses = {}
        
        def get(self, url: str) -> dict[str, Any]:
            return self.responses.get(url, {"status": "not_found"})
        
        def post(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
            return {"status": "created", "data": data}
        
        def set_response(self, url: str, response: dict[str, Any]):
            self.responses[url] = response
    
    return MockAPIClient()

# Test data factories
class UserFactory:
    """Factory for creating test users."""
    
    @staticmethod
    def create_user(name: str = None, email: str = None, **kwargs) -> dict[str, Any]:
        """Create test user with defaults."""
        import random
        import string
        
        if name is None:
            name = f"User {''.join(random.choices(string.ascii_letters, k=8))}"
        
        if email is None:
            username = name.lower().replace(" ", ".")
            email = f"{username}@test.example.com"
        
        user = {
            "id": random.randint(1, 10000),
            "name": name,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            **kwargs
        }
        return user
    
    @staticmethod
    def create_users(count: int) -> list[dict[str, Any]]:
        """Create multiple test users."""
        return [UserFactory.create_user() for _ in range(count)]

@pytest.fixture
def user_factory():
    """User factory fixture."""
    return UserFactory()

# Custom assertions
def assert_valid_user_response(response: dict[str, Any], expected_user: dict[str, Any]) -> None:
    """Assert that API response contains valid user data."""
    assert "data" in response
    user_data = response["data"]
    
    assert user_data["id"] == expected_user["id"]
    assert user_data["name"] == expected_user["name"]
    assert user_data["email"] == expected_user["email"]
    assert "created_at" in user_data

def assert_valid_timestamp(timestamp_str: str, tolerance_seconds: int = 60) -> None:
    """Assert timestamp is recent and valid format."""
    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now()
        diff = abs((now - timestamp).total_seconds())
        assert diff <= tolerance_seconds, f"Timestamp {timestamp_str} is not recent enough"
    except ValueError:
        pytest.fail(f"Invalid timestamp format: {timestamp_str}")

# Parametrized test generators
def generate_email_test_cases():
    """Generate email validation test cases."""
    valid_emails = [
        "user@example.com",
        "test.user+tag@domain.co.uk",
        "user123@test-domain.com"
    ]
    
    invalid_emails = [
        "invalid.email",
        "@domain.com",
        "user@",
        "user@@domain.com"
    ]
    
    cases = []
    for email in valid_emails:
        cases.append((email, True))
    for email in invalid_emails:
        cases.append((email, False))
    
    return cases

# Custom markers for test categorization
pytestmark = [
    pytest.mark.api_test,
    pytest.mark.integration
]

class TestFrameworkExtensions:
    """Advanced testing patterns and extensions."""
    
    @pytest.mark.api_test
    def test_api_endpoint_with_factory(self, api_client, user_factory):
        """Test API endpoint using factory and custom assertions."""
        user = user_factory.create_user()
        
        # Setup API response
        api_client.set_response(f"/users/{user['id']}", {"data": user})
        
        # Test API call
        response = api_client.get(f"/users/{user['id']}")
        
        # Use custom assertion
        assert_valid_user_response(response, user)
    
    @pytest.mark.parametrize("email,expected", generate_email_test_cases())
    def test_generated_email_cases(self, email, expected):
        """Test with dynamically generated test cases."""
        result = validate_email(email)
        assert result == expected
    
    def test_custom_timestamp_assertion(self):
        """Test custom timestamp assertion helper."""
        current_time = datetime.now().isoformat()
        assert_valid_timestamp(current_time)
        
        # Test with old timestamp (should fail)
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        with pytest.raises(AssertionError, match="not recent enough"):
            assert_valid_timestamp(old_time, tolerance_seconds=30)

# =============================================================================
# COMPLETE TEST CONFIGURATION
# =============================================================================

# pytest configuration (would go in pytest.ini)
PYTEST_CONFIG = """
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py xample.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --tb=short
    -v
    --durations=10
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    api_test: API integration tests
    performance: Performance tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
"""

# Test helper utilities
class TestHelpers:
    """Collection of test helper utilities."""
    
    @staticmethod
    def create_temp_file(content: str, suffix: str = ".txt") -> str:
        """Create temporary file with content."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
    
    @staticmethod
    def assert_files_equal(file1: str, file2: str) -> None:
        """Assert two files have identical content."""
        with open(file1) as f1, open(file2) as f2:
            assert f1.read() == f2.read()
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> tuple:
        """Measure function execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration

# =============================================================================
# TEST RUNNER AND FINAL CONFIGURATION
# =============================================================================

def run_specific_tests(test_pattern: str = None, markers: str = None):
    """Run specific tests based on pattern or markers."""
    args = [__file__, "-v"]
    
    if test_pattern:
        args.extend(["-k", test_pattern])
    
    if markers:
        args.extend(["-m", markers])
    
    return pytest.main(args)

def run_performance_tests():
    """Run only performance tests."""
    return run_specific_tests(markers="slow or performance")

def run_unit_tests():
    """Run only unit tests."""
    return run_specific_tests(markers="unit")

def run_integration_tests():
    """Run only integration tests."""
    return run_specific_tests(markers="integration")

if __name__ == "__main__":
    # Run all tests by default
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "unit":
            exit_code = run_unit_tests()
        elif test_type == "integration":
            exit_code = run_integration_tests()
        elif test_type == "performance":
            exit_code = run_performance_tests()
        else:
            exit_code = run_specific_tests(test_pattern=test_type)
    else:
        exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    sys.exit(exit_code)
