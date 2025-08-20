"""
Domain-Driven Design (DDD) Workshop - Python Implementation Patterns

This workshop covers Domain-Driven Design patterns implemented in Python,
showing Java/C# developers how to apply DDD principles using Python's
unique features like dataclasses, properties, and dynamic typing.

Complete the following tasks to master DDD patterns in Python.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar, Protocol
from datetime import datetime, timezone
from uuid import UUID, uuid4
from enum import Enum
import weakref
from collections import defaultdict
import asyncio

# =============================================================================
# TASK 1: Value Objects and Entity Patterns
# =============================================================================

"""
TASK 1: Implement Value Objects and Entities

In DDD, Value Objects are immutable objects defined by their attributes,
while Entities have identity and lifecycle. Python's dataclasses make
implementing these patterns more concise than Java/C#.

Requirements:
- Immutable Value Objects with validation
- Entities with identity and equality based on ID
- Rich domain models with behavior
- Type safety and validation
- Comparison and hashing for Value Objects

Example usage:
email = EmailAddress("john@example.com")  # Value Object
user = User(user_id=UserId(), email=email)  # Entity
order = Order.create_new(user_id=user.id, items=[...])
"""

@dataclass(frozen=True)
class ValueObject:
    """Base class for Value Objects - immutable by design"""
    
    def __post_init__(self):
        """Override to add validation logic"""
        self.validate()
    
    def validate(self) -> None:
        """Validate the value object - override in subclasses"""
        pass

@dataclass(frozen=True)
class EmailAddress(ValueObject):
    """Email address value object with validation"""
    value: str
    
    def validate(self) -> None:
        # Your implementation here
        pass
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def domain(self) -> str:
        """Extract domain from email"""
        # Your implementation here
        pass

@dataclass(frozen=True)
class Money(ValueObject):
    """Money value object with currency"""
    amount: float
    currency: str = "USD"
    
    def validate(self) -> None:
        # Your implementation here
        pass
    
    def add(self, other: "Money") -> "Money":
        """Add two money objects"""
        # Your implementation here
        pass
    
    def multiply(self, factor: float) -> "Money":
        """Multiply money by factor"""
        # Your implementation here
        pass

@dataclass(frozen=True)
class UserId(ValueObject):
    """Strongly typed user identifier"""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)

@dataclass(frozen=True)
class ProductId(ValueObject):
    """Strongly typed product identifier"""
    value: UUID = field(default_factory=uuid4)

class Entity(ABC):
    """Base class for Domain Entities"""
    
    def __init__(self, entity_id: ValueObject):
        # Your implementation here
        pass
    
    @property
    def id(self) -> ValueObject:
        """Entity identifier"""
        # Your implementation here
        pass
    
    def __eq__(self, other) -> bool:
        """Entities are equal if they have the same ID and type"""
        # Your implementation here
        pass
    
    def __hash__(self) -> int:
        """Hash based on ID"""
        # Your implementation here
        pass

class User(Entity):
    """User entity with rich domain behavior"""
    
    def __init__(self, user_id: UserId, email: EmailAddress, name: str):
        # Your implementation here
        pass
    
    def change_email(self, new_email: EmailAddress) -> None:
        """Change user email with validation"""
        # Your implementation here
        pass
    
    def is_active(self) -> bool:
        """Check if user is active"""
        # Your implementation here
        pass
    
    def deactivate(self) -> None:
        """Deactivate user account"""
        # Your implementation here
        pass

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Order(Entity):
    """Order entity with business logic"""
    
    def __init__(self, order_id: ValueObject, user_id: UserId):
        # Your implementation here
        pass
    
    @classmethod
    def create_new(cls, user_id: UserId) -> "Order":
        """Factory method to create new order"""
        # Your implementation here
        pass
    
    def add_item(self, product_id: ProductId, quantity: int, price: Money) -> None:
        """Add item to order"""
        # Your implementation here
        pass
    
    def calculate_total(self) -> Money:
        """Calculate order total"""
        # Your implementation here
        pass
    
    def confirm(self) -> None:
        """Confirm the order"""
        # Your implementation here
        pass
    
    def can_be_cancelled(self) -> bool:
        """Check if order can be cancelled"""
        # Your implementation here
        pass

# =============================================================================
# TASK 2: Aggregate Roots and Domain Services
# =============================================================================

"""
TASK 2: Implement Aggregates and Domain Services

Aggregates enforce business invariants and provide consistency boundaries.
Domain Services contain business logic that doesn't belong to any single entity.

Requirements:
- Aggregate roots with invariant enforcement
- Domain services for cross-entity business logic
- Proper encapsulation and consistency boundaries
- Domain events for aggregate state changes
- Transaction boundaries

Example usage:
user_service = UserRegistrationService(user_repo, email_service)
user = user_service.register_user(email, password)

order_aggregate = OrderAggregate.create(user_id)
order_aggregate.add_item(product_id, quantity, price)
"""

class DomainEvent:
    """Base class for domain events"""
    
    def __init__(self, aggregate_id: ValueObject, event_data: dict[str, Any]):
        # Your implementation here
        pass
    
    @property
    def occurred_on(self) -> datetime:
        # Your implementation here
        pass

class UserRegisteredEvent(DomainEvent):
    """Event raised when user is registered"""
    
    def __init__(self, user_id: UserId, email: EmailAddress):
        # Your implementation here
        pass

class OrderConfirmedEvent(DomainEvent):
    """Event raised when order is confirmed"""
    
    def __init__(self, order_id: ValueObject, user_id: UserId, total: Money):
        # Your implementation here
        pass

class AggregateRoot(Entity):
    """Base class for Aggregate Roots"""
    
    def __init__(self, entity_id: ValueObject):
        # Your implementation here
        pass
    
    def add_domain_event(self, event: DomainEvent) -> None:
        """Add domain event to be published"""
        # Your implementation here
        pass
    
    def clear_domain_events(self) -> list[DomainEvent]:
        """Clear and return domain events"""
        # Your implementation here
        pass
    
    def get_uncommitted_events(self) -> list[DomainEvent]:
        """Get events that haven't been published yet"""
        # Your implementation here
        pass

class UserAggregate(AggregateRoot):
    """User aggregate root"""
    
    def __init__(self, user_id: UserId, email: EmailAddress, name: str):
        # Your implementation here
        pass
    
    @classmethod
    def register_new_user(cls, email: EmailAddress, name: str) -> "UserAggregate":
        """Factory method for user registration"""
        # Your implementation here
        pass
    
    def change_email(self, new_email: EmailAddress) -> None:
        """Change email with domain event"""
        # Your implementation here
        pass

class OrderAggregate(AggregateRoot):
    """Order aggregate root with business invariants"""
    
    def __init__(self, order_id: ValueObject, user_id: UserId):
        # Your implementation here
        pass
    
    @classmethod
    def create(cls, user_id: UserId) -> "OrderAggregate":
        """Create new order aggregate"""
        # Your implementation here
        pass
    
    def add_item(self, product_id: ProductId, quantity: int, price: Money) -> None:
        """Add item with business rule validation"""
        # Your implementation here
        pass
    
    def confirm_order(self) -> None:
        """Confirm order with business rules"""
        # Your implementation here
        pass

class DomainService(ABC):
    """Base class for Domain Services"""
    pass

class UserRegistrationService(DomainService):
    """Domain service for user registration business logic"""
    
    def __init__(self, user_repository, email_uniqueness_service):
        # Your implementation here
        pass
    
    async def register_user(self, email: EmailAddress, name: str) -> UserAggregate:
        """Register new user with business rules"""
        # Your implementation here
        pass
    
    async def is_email_available(self, email: EmailAddress) -> bool:
        """Check if email is available"""
        # Your implementation here
        pass

class OrderPricingService(DomainService):
    """Domain service for order pricing logic"""
    
    def __init__(self, pricing_repository):
        # Your implementation here
        pass
    
    async def calculate_order_total(self, order: OrderAggregate) -> Money:
        """Calculate order total with discounts and taxes"""
        # Your implementation here
        pass
    
    async def apply_discount(self, order: OrderAggregate, 
                           discount_code: str) -> Money:
        """Apply discount to order"""
        # Your implementation here
        pass

# =============================================================================
# TASK 3: Repository Patterns with Python Specifics
# =============================================================================

"""
TASK 3: Implement Repository Pattern

Repositories provide an abstraction over data access. Python's async/await
and type hints enable clean repository implementations.

Requirements:
- Generic repository base class
- Async repository operations
- Specification pattern for queries
- Unit of Work pattern for transactions
- In-memory repositories for testing

Example usage:
async with unit_of_work:
    user = await user_repository.get_by_id(user_id)
    user.change_email(new_email)
    await user_repository.save(user)
    await unit_of_work.commit()
"""

T = TypeVar('T', bound=Entity)

class Repository(ABC, Generic[T]):
    """Generic repository interface"""
    
    @abstractmethod
    async def get_by_id(self, entity_id: ValueObject) -> T | None:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> None:
        """Save entity"""
        pass
    
    @abstractmethod
    async def delete(self, entity: T) -> None:
        """Delete entity"""
        pass
    
    @abstractmethod
    async def find_by_specification(self, spec: "Specification") -> list[T]:
        """Find entities matching specification"""
        pass

class Specification(ABC):
    """Specification pattern for queries"""
    
    @abstractmethod
    def is_satisfied_by(self, entity: Entity) -> bool:
        """Check if entity satisfies specification"""
        pass
    
    def and_specification(self, other: "Specification") -> "AndSpecification":
        """Combine specifications with AND"""
        return AndSpecification(self, other)
    
    def or_specification(self, other: "Specification") -> "OrSpecification":
        """Combine specifications with OR"""
        return OrSpecification(self, other)

class AndSpecification(Specification):
    """AND combination of specifications"""
    
    def __init__(self, left: Specification, right: Specification):
        # Your implementation here
        pass
    
    def is_satisfied_by(self, entity: Entity) -> bool:
        # Your implementation here
        pass

class OrSpecification(Specification):
    """OR combination of specifications"""
    
    def __init__(self, left: Specification, right: Specification):
        # Your implementation here
        pass

class UserByEmailSpecification(Specification):
    """Specification to find user by email"""
    
    def __init__(self, email: EmailAddress):
        # Your implementation here
        pass
    
    def is_satisfied_by(self, entity: Entity) -> bool:
        # Your implementation here
        pass

class ActiveUsersSpecification(Specification):
    """Specification for active users"""
    
    def is_satisfied_by(self, entity: Entity) -> bool:
        # Your implementation here
        pass

class UserRepository(Repository[UserAggregate]):
    """User repository interface"""
    
    @abstractmethod
    async def find_by_email(self, email: EmailAddress) -> UserAggregate | None:
        """Find user by email"""
        pass
    
    @abstractmethod
    async def get_active_users(self) -> list[UserAggregate]:
        """Get all active users"""
        pass

class InMemoryUserRepository(UserRepository):
    """In-memory user repository for testing"""
    
    def __init__(self):
        # Your implementation here
        pass
    
    async def get_by_id(self, user_id: UserId) -> UserAggregate | None:
        # Your implementation here
        pass
    
    async def save(self, user: UserAggregate) -> None:
        # Your implementation here
        pass
    
    async def delete(self, user: UserAggregate) -> None:
        # Your implementation here
        pass
    
    async def find_by_email(self, email: EmailAddress) -> UserAggregate | None:
        # Your implementation here
        pass

class UnitOfWork(ABC):
    """Unit of Work pattern for transaction management"""
    
    @abstractmethod
    async def __aenter__(self):
        """Start transaction"""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End transaction"""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit transaction"""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback transaction"""
        pass

class InMemoryUnitOfWork(UnitOfWork):
    """In-memory unit of work for testing"""
    
    def __init__(self):
        # Your implementation here
        pass

# =============================================================================
# TASK 4: Domain Events and Event Sourcing
# =============================================================================

"""
TASK 4: Implement Domain Events and Event Sourcing

Domain events capture important business events. Event sourcing stores
all changes as a sequence of events rather than current state.

Requirements:
- Domain event dispatcher
- Event store for persistence
- Event sourcing for aggregates
- Event replay and projection
- Saga pattern for process management

Example usage:
event_store = EventStore()
await event_store.save_events(aggregate_id, events)
events = await event_store.get_events(aggregate_id)
aggregate = UserAggregate.from_events(events)
"""

class EventStore(ABC):
    """Abstract event store"""
    
    @abstractmethod
    async def save_events(self, aggregate_id: ValueObject, 
                         events: list[DomainEvent], 
                         expected_version: int = -1) -> None:
        """Save events for aggregate"""
        pass
    
    @abstractmethod
    async def get_events(self, aggregate_id: ValueObject, 
                        from_version: int = 0) -> list[DomainEvent]:
        """Get events for aggregate"""
        pass
    
    @abstractmethod
    async def get_all_events(self, from_timestamp: datetime = None) -> list[DomainEvent]:
        """Get all events from timestamp"""
        pass

class InMemoryEventStore(EventStore):
    """In-memory event store for testing"""
    
    def __init__(self):
        # Your implementation here
        pass
    
    async def save_events(self, aggregate_id: ValueObject, 
                         events: list[DomainEvent], 
                         expected_version: int = -1) -> None:
        # Your implementation here
        pass
    
    async def get_events(self, aggregate_id: ValueObject, 
                        from_version: int = 0) -> list[DomainEvent]:
        # Your implementation here
        pass

class EventSourcedAggregate(AggregateRoot):
    """Base class for event-sourced aggregates"""
    
    def __init__(self, aggregate_id: ValueObject):
        # Your implementation here
        pass
    
    @classmethod
    def from_events(cls, events: list[DomainEvent]) -> "EventSourcedAggregate":
        """Reconstruct aggregate from events"""
        # Your implementation here
        pass
    
    def apply_event(self, event: DomainEvent) -> None:
        """Apply event to aggregate state"""
        # Your implementation here
        pass
    
    def get_changes(self) -> list[DomainEvent]:
        """Get uncommitted changes"""
        # Your implementation here
        pass

class EventDispatcher:
    """Dispatch domain events to handlers"""
    
    def __init__(self):
        # Your implementation here
        pass
    
    def subscribe(self, event_type: type, handler: callable) -> None:
        """Subscribe handler to event type"""
        # Your implementation here
        pass
    
    async def dispatch(self, events: list[DomainEvent]) -> None:
        """Dispatch events to handlers"""
        # Your implementation here
        pass

class EventHandler(ABC):
    """Base class for event handlers"""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event"""
        pass

class UserRegistrationEmailHandler(EventHandler):
    """Send welcome email when user registers"""
    
    def __init__(self, email_service):
        # Your implementation here
        pass
    
    async def handle(self, event: UserRegisteredEvent) -> None:
        """Send welcome email"""
        # Your implementation here
        pass

# =============================================================================
# TASK 5: CQRS (Command Query Responsibility Segregation)
# =============================================================================

"""
TASK 5: Implement CQRS Pattern

CQRS separates read and write operations, allowing for optimized
data models for each use case.

Requirements:
- Command and Query separation
- Command handlers for write operations
- Query handlers for read operations
- Read models optimized for queries
- Event-driven read model updates

Example usage:
command_bus = CommandBus()
command_bus.register_handler(RegisterUserCommand, RegisterUserHandler())

query_bus = QueryBus()
query_bus.register_handler(GetUserQuery, GetUserHandler())

await command_bus.execute(RegisterUserCommand(email, name))
user_data = await query_bus.execute(GetUserQuery(user_id))
"""

class Command(ABC):
    """Base class for commands"""
    pass

class Query(ABC):
    """Base class for queries"""
    pass

class RegisterUserCommand(Command):
    """Command to register new user"""
    
    def __init__(self, email: EmailAddress, name: str):
        # Your implementation here
        pass

class UpdateUserEmailCommand(Command):
    """Command to update user email"""
    
    def __init__(self, user_id: UserId, new_email: EmailAddress):
        # Your implementation here
        pass

class GetUserQuery(Query):
    """Query to get user by ID"""
    
    def __init__(self, user_id: UserId):
        # Your implementation here
        pass

class GetActiveUsersQuery(Query):
    """Query to get all active users"""
    pass

class CommandHandler(ABC):
    """Base class for command handlers"""
    
    @abstractmethod
    async def handle(self, command: Command) -> Any:
        """Handle command"""
        pass

class QueryHandler(ABC):
    """Base class for query handlers"""
    
    @abstractmethod
    async def handle(self, query: Query) -> Any:
        """Handle query"""
        pass

class RegisterUserHandler(CommandHandler):
    """Handler for user registration command"""
    
    def __init__(self, user_repository: UserRepository, 
                 registration_service: UserRegistrationService):
        # Your implementation here
        pass
    
    async def handle(self, command: RegisterUserCommand) -> UserId:
        """Handle user registration"""
        # Your implementation here
        pass

class GetUserHandler(QueryHandler):
    """Handler for get user query"""
    
    def __init__(self, user_read_model: "UserReadModel"):
        # Your implementation here
        pass
    
    async def handle(self, query: GetUserQuery) -> dict[str, Any | None]:
        """Handle get user query"""
        # Your implementation here
        pass

class CommandBus:
    """Command bus for executing commands"""
    
    def __init__(self):
        # Your implementation here
        pass
    
    def register_handler(self, command_type: type, handler: CommandHandler) -> None:
        """Register command handler"""
        # Your implementation here
        pass
    
    async def execute(self, command: Command) -> Any:
        """Execute command"""
        # Your implementation here
        pass

class QueryBus:
    """Query bus for executing queries"""
    
    def __init__(self):
        # Your implementation here
        pass
    
    def register_handler(self, query_type: type, handler: QueryHandler) -> None:
        """Register query handler"""
        # Your implementation here
        pass
    
    async def execute(self, query: Query) -> Any:
        """Execute query"""
        # Your implementation here
        pass

class UserReadModel:
    """Read model optimized for user queries"""
    
    def __init__(self):
        # Your implementation here
        pass
    
    async def get_by_id(self, user_id: UserId) -> dict[str, Any | None]:
        """Get user data optimized for reading"""
        # Your implementation here
        pass
    
    async def get_active_users(self) -> list[dict[str, Any]]:
        """Get active users optimized for reading"""
        # Your implementation here
        pass
    
    async def update_from_event(self, event: DomainEvent) -> None:
        """Update read model from domain event"""
        # Your implementation here
        pass

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_domain_driven_design():
    """Test all DDD pattern implementations"""
    print("Testing Domain-Driven Design Patterns...")
    
    # Test Value Objects and Entities
    print("\n1. Testing Value Objects and Entities:")
    # Your test implementation here
    
    # Test Aggregates and Domain Services
    print("\n2. Testing Aggregates and Domain Services:")
    # Your test implementation here
    
    # Test Repository Pattern
    print("\n3. Testing Repository Pattern:")
    # Your test implementation here
    
    # Test Domain Events
    print("\n4. Testing Domain Events:")
    # Your test implementation here
    
    # Test CQRS Pattern
    print("\n5. Testing CQRS Pattern:")
    # Your test implementation here
    
    print("\n✅ All DDD pattern tests completed!")

if __name__ == "__main__":
    test_domain_driven_design()
