SPEC_ID: b11554d2-9748-4ad5-adb5-6ce59c430ece
SPEC_HASH: 164425f26c0c0decb954a731ebab900252b3e7e3242521324ee0ed8fab5084d4

# Orb Repository Architecture Map

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Orb Billing Platform                     │
├─────────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)           │  Core Business Logic           │
│  - REST Endpoints              │  - Billing Engine              │
│  - Authentication              │  - Usage Aggregation           │
│  - Request/Response Handling   │  - Invoice Generation          │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                    │  External Integrations         │
│  - PostgreSQL (Primary)        │  - Stripe (Payments)           │
│  - Redis (Caching)             │  - Webhooks                    │
│  - ClickHouse (Analytics)      │  - Third-party APIs            │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure & Key Components

### `/orb/` - Core Application Package

```
orb/
├── __init__.py
├── app.py                      # FastAPI application factory
├── config.py                   # Configuration management
├── constants.py                # System-wide constants
├── exceptions.py               # Custom exception classes
└── middleware/                 # Request/response middleware
    ├── __init__.py
    ├── auth.py                # Authentication middleware
    ├── cors.py                # CORS handling
    └── logging.py             # Request logging
```

**Key Decisions:**
- Centralized configuration in `config.py` for environment management
- Custom exceptions for consistent error handling across the platform
- Middleware-based architecture for cross-cutting concerns

### `/orb/billing/` - Core Billing Engine

```
orb/billing/
├── __init__.py
├── models/                     # Data models
│   ├── __init__.py
│   ├── customer.py            # Customer entity
│   ├── subscription.py        # Subscription management
│   ├── plan.py               # Billing plans
│   ├── invoice.py            # Invoice generation
│   └── usage.py              # Usage tracking
├── services/                   # Business logic layer
│   ├── __init__.py
│   ├── billing_engine.py     # Core billing calculations
│   ├── usage_aggregator.py   # Usage data aggregation
│   ├── invoice_generator.py  # Invoice creation logic
│   └── subscription_manager.py # Subscription lifecycle
└── schemas/                    # Pydantic schemas
    ├── __init__.py
    ├── customer.py
    ├── subscription.py
    └── invoice.py
```

**Architecture Decisions:**
- **Models-Services-Schemas separation**: Clean separation of concerns
- **Domain-driven design**: Each billing concept has its own module
- **Pydantic schemas**: Type-safe API contracts and validation

### `/orb/api/` - API Layer

```
orb/api/
├── __init__.py
├── v1/                        # API versioning
│   ├── __init__.py
│   ├── router.py             # Main API router
│   ├── endpoints/            # REST endpoints
│   │   ├── __init__.py
│   │   ├── customers.py      # Customer management
│   │   ├── subscriptions.py  # Subscription operations
│   │   ├── invoices.py       # Invoice endpoints
│   │   ├── usage.py          # Usage reporting
│   │   └── webhooks.py       # Webhook handlers
│   └── dependencies.py       # FastAPI dependencies
└── auth/                      # Authentication
    ├── __init__.py
    ├── jwt.py                # JWT token handling
    └── permissions.py        # Role-based access control
```

**Key Features:**
- **API Versioning**: `/v1/` structure allows for future API evolution
- **Dependency Injection**: FastAPI dependencies for auth, DB connections
- **Webhook Support**: Dedicated webhook handling for external integrations

### `/orb/database/` - Data Layer

```
orb/database/
├── __init__.py
├── connection.py              # Database connection management
├── migrations/                # Alembic migrations
│   ├── env.py
│   ├── script.py.mako
│   └── versions/             # Migration files
│       └── *.py
├── repositories/              # Data access layer
│   ├── __init__.py
│   ├── base.py               # Base repository pattern
│   ├── customer.py           # Customer data access
│   ├── subscription.py       # Subscription data access
│   └── invoice.py            # Invoice data access
└── models/                    # SQLAlchemy ORM models
    ├── __init__.py
    ├── base.py               # Base model class
    ├── customer.py
    ├── subscription.py
    └── invoice.py
```

**Data Architecture Decisions:**
- **Repository Pattern**: Abstraction layer over database operations
- **Alembic Migrations**: Version-controlled database schema changes
- **SQLAlchemy ORM**: Type-safe database interactions with relationship mapping

### `/orb/integrations/` - External Service Integrations

```
orb/integrations/
├── __init__.py
├── stripe/                    # Stripe payment processing
│   ├── __init__.py
│   ├── client.py             # Stripe API client
│   ├── webhooks.py           # Stripe webhook handling
│   └── models.py             # Stripe data models
├── analytics/                 # Analytics integrations
│   ├── __init__.py
│   ├── clickhouse.py         # ClickHouse client
│   └── usage_tracker.py      # Usage event tracking
└── notifications/             # Communication services
    ├── __init__.py
    ├── email.py              # Email service integration
    └── slack.py              # Slack notifications
```

**Integration Strategy:**
- **Client Pattern**: Dedicated clients for each external service
- **Webhook Isolation**: Separate webhook handling per integration
- **Analytics Pipeline**: ClickHouse for high-volume usage data

### `/orb/tasks/` - Background Job Processing

```
orb/tasks/
├── __init__.py
├── celery_app.py             # Celery configuration
├── billing/                  # Billing-related tasks
│   ├── __init__.py
│   ├── invoice_generation.py # Async invoice creation
│   ├── usage_aggregation.py  # Periodic usage rollups
│   └── subscription_renewal.py # Subscription lifecycle
└── notifications/            # Notification tasks
    ├── __init__.py
    └── email_sender.py       # Async email delivery
```

**Background Processing Design:**
- **Celery Integration**: Distributed task queue for async operations
- **Domain Separation**: Tasks organized by business domain
- **Reliability**: Retry mechanisms and error handling for critical operations

### `/tests/` - Test Suite

```
tests/
├── __init__.py
├── conftest.py               # Pytest configuration
├── unit/                     # Unit tests
│   ├── billing/
│   ├── api/
│   └── database/
├── integration/              # Integration tests
│   ├── test_billing_flow.py
│   └── test_api_endpoints.py
└── fixtures/                 # Test data
    ├── customers.json
    └── subscriptions.json
```

### Root Configuration Files

```
/
├── pyproject.toml            # Python project configuration
├── docker-compose.yml        # Local development environment
├── Dockerfile               # Container image definition
├── alembic.ini              # Database migration configuration
├── .env.example             # Environment variables template
└── requirements/            # Dependency management
    ├── base.txt
    ├── dev.txt
    └── prod.txt
```

## Key Architectural Patterns

### 1. **Layered Architecture**
- **API Layer**: FastAPI endpoints and request handling
- **Service Layer**: Business logic and domain operations
- **Data Layer**: Repository pattern with SQLAlchemy ORM
- **Integration Layer**: External service abstractions

### 2. **Domain-Driven Design**
- Core billing concepts (Customer, Subscription, Invoice) as first-class entities
- Services encapsulate business rules and workflows
- Clear boundaries between billing, payments, and analytics domains

### 3. **Dependency Injection**
- FastAPI's dependency system for database connections, authentication
- Repository injection for testability and flexibility
- Configuration injection for environment-specific behavior

### 4. **Event-Driven Architecture**
- Webhook handlers for external system events
- Background tasks for async processing
- Usage events feeding into analytics pipeline

## Scalability Considerations

### **Database Strategy**
- **PostgreSQL**: ACID compliance for billing data integrity
- **Redis**: Caching layer for frequently accessed data
- **ClickHouse**: Analytics database for usage aggregation at scale

### **Horizontal Scaling**
- Stateless API design enables horizontal scaling
- Background task workers can be scaled independently
- Database read replicas for query performance

### **Performance Optimizations**
- Repository pattern enables efficient query optimization
- Caching strategies in Redis for hot data paths
- Async processing for non-blocking operations

This architecture provides a solid foundation for a billing platform with clear separation of concerns, scalability built-in, and maintainable code organization.