SPEC_ID: 6628b9e1-8b79-48d4-a864-b1f45c7d9b71
SPEC_HASH: 3e3c48f4317e719feec1f189298f7b726081ab311dd86a82d1760eb2979d3d16

# Orb Repository Architecture Map

## System Overview
Orb is a billing and revenue management platform built with a modern Python stack, featuring a Django-based API backend with comprehensive billing capabilities, subscription management, and usage-based pricing models.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orb Platform                              │
├─────────────────┬───────────────────┬───────────────────────┤
│   API Layer     │   Business Logic  │   Data & Integration  │
│                 │                   │                       │
│ • REST API      │ • Billing Engine  │ • PostgreSQL         │
│ • GraphQL       │ • Subscription    │ • Redis Cache        │
│ • WebSockets    │ • Usage Tracking  │ • External APIs      │
│ • Authentication│ • Pricing Models  │ • Event Streaming    │
└─────────────────┴───────────────────┴───────────────────────┘
```

## Directory Structure & Key Components

### Core Application (`/orb/`)
```
orb/
├── settings/                    # Django configuration
│   ├── base.py                 # Base settings
│   ├── development.py          # Dev environment
│   ├── production.py           # Prod environment
│   └── testing.py              # Test configuration
├── urls.py                     # Root URL configuration
├── wsgi.py                     # WSGI application
├── asgi.py                     # ASGI for async support
└── celery.py                   # Celery task queue setup
```

### API Layer (`/api/`)
```
api/
├── v1/                         # API versioning
│   ├── serializers/           # DRF serializers
│   ├── views/                 # API endpoints
│   ├── permissions.py         # Access control
│   └── urls.py               # API routing
├── middleware/                # Custom middleware
├── authentication/           # Auth backends
└── exceptions.py            # Custom exception handlers
```

### Billing Engine (`/billing/`)
```
billing/
├── models/                    # Core billing models
│   ├── customer.py           # Customer management
│   ├── subscription.py       # Subscription lifecycle
│   ├── plan.py              # Pricing plans
│   ├── usage.py             # Usage tracking
│   ├── invoice.py           # Invoice generation
│   └── payment.py           # Payment processing
├── services/                 # Business logic services
│   ├── billing_service.py   # Core billing operations
│   ├── usage_aggregator.py  # Usage calculation
│   ├── invoice_generator.py # Invoice creation
│   └── pricing_engine.py    # Price calculations
├── tasks/                   # Async tasks
│   ├── invoice_tasks.py     # Invoice generation
│   ├── usage_tasks.py       # Usage processing
│   └── payment_tasks.py     # Payment processing
└── utils/                   # Billing utilities
```

### Subscription Management (`/subscriptions/`)
```
subscriptions/
├── models/
│   ├── subscription.py      # Subscription state
│   ├── plan_version.py      # Plan versioning
│   └── addon.py            # Add-on products
├── services/
│   ├── subscription_service.py  # Lifecycle management
│   ├── plan_service.py         # Plan operations
│   └── upgrade_service.py      # Plan changes
└── state_machine/              # Subscription states
    ├── states.py              # State definitions
    └── transitions.py         # State transitions
```

### Usage Tracking (`/usage/`)
```
usage/
├── models/
│   ├── event.py             # Usage events
│   ├── metric.py            # Billable metrics
│   └── aggregation.py       # Usage aggregations
├── collectors/              # Data ingestion
│   ├── api_collector.py     # API usage collection
│   ├── webhook_collector.py # Webhook ingestion
│   └── batch_collector.py   # Batch processing
├── processors/              # Event processing
│   ├── event_processor.py   # Real-time processing
│   ├── aggregator.py        # Usage aggregation
│   └── deduplicator.py      # Event deduplication
└── storage/                 # Storage backends
    ├── timeseries.py        # Time-series data
    └── warehouse.py         # Data warehouse
```

### Customer Management (`/customers/`)
```
customers/
├── models/
│   ├── customer.py          # Customer profiles
│   ├── contact.py           # Contact information
│   └── organization.py      # Multi-tenant support
├── services/
│   ├── customer_service.py  # Customer operations
│   ├── onboarding.py        # Customer onboarding
│   └── segmentation.py      # Customer segments
└── integrations/            # CRM integrations
```

### Payment Processing (`/payments/`)
```
payments/
├── models/
│   ├── payment_method.py    # Payment methods
│   ├── transaction.py       # Payment transactions
│   └── refund.py           # Refund handling
├── processors/              # Payment gateways
│   ├── stripe_processor.py  # Stripe integration
│   ├── braintree_processor.py # Braintree integration
│   └── base_processor.py    # Base processor class
├── services/
│   ├── payment_service.py   # Payment orchestration
│   ├── retry_service.py     # Failed payment retry
│   └── reconciliation.py   # Payment reconciliation
└── webhooks/               # Payment webhooks
```

### Reporting & Analytics (`/analytics/`)
```
analytics/
├── models/
│   ├── report.py           # Report definitions
│   └── dashboard.py        # Dashboard configs
├── engines/
│   ├── revenue_engine.py   # Revenue analytics
│   ├── usage_engine.py     # Usage analytics
│   └── cohort_engine.py    # Cohort analysis
├── exporters/
│   ├── csv_exporter.py     # CSV export
│   ├── pdf_exporter.py     # PDF reports
│   └── api_exporter.py     # API data export
└── schedulers/             # Report scheduling
```

### Infrastructure & Utilities

#### Database Layer (`/db/`)
```
db/
├── migrations/             # Database migrations
├── seeds/                 # Test data seeds
├── indexes/               # Database indexes
└── constraints/           # Database constraints
```

#### Caching Layer (`/cache/`)
```
cache/
├── redis_client.py        # Redis connection
├── cache_keys.py          # Cache key management
├── invalidation.py        # Cache invalidation
└── warming.py             # Cache warming strategies
```

#### Task Queue (`/tasks/`)
```
tasks/
├── celery_app.py          # Celery application
├── beat_schedule.py       # Periodic tasks
├── workers/               # Task workers
└── monitoring.py          # Task monitoring
```

#### External Integrations (`/integrations/`)
```
integrations/
├── stripe/                # Stripe payment gateway
├── salesforce/           # CRM integration
├── hubspot/              # Marketing automation
├── segment/              # Analytics platform
└── webhooks/             # Webhook handlers
```

## Key Architectural Decisions

### 1. **Modular Monolith Architecture**
- **Decision**: Single Django application with clear module boundaries
- **Rationale**: Easier development and deployment while maintaining separation of concerns
- **Trade-offs**: Potential scaling limitations vs. development velocity

### 2. **Event-Driven Usage Tracking**
- **Decision**: Async event processing with Celery + Redis
- **Rationale**: Handle high-volume usage data without blocking API responses
- **Implementation**: `usage/collectors/` → `usage/processors/` → `billing/services/`

### 3. **Multi-Tenant Data Model**
- **Decision**: Shared database with tenant isolation via foreign keys
- **Rationale**: Cost-effective scaling with strong data isolation
- **Security**: Row-level security policies in PostgreSQL

### 4. **Pluggable Payment Processors**
- **Decision**: Abstract payment processor interface
- **Rationale**: Support multiple payment gateways without tight coupling
- **Extension**: Easy to add new processors via `payments/processors/base_processor.py`

### 5. **State Machine for Subscriptions**
- **Decision**: Explicit state management for subscription lifecycle
- **Rationale**: Complex billing rules require predictable state transitions
- **Implementation**: `subscriptions/state_machine/` handles all transitions

## Data Flow Architecture

### Usage Event Processing
```
API/Webhook → Event Queue → Processor → Aggregator → Billing Engine
     ↓              ↓            ↓           ↓            ↓
usage/collectors → tasks/queue → usage/processors → billing/services
```

### Invoice Generation Flow
```
Subscription → Usage Data → Pricing Engine → Invoice → Payment
     ↓              ↓            ↓            ↓         ↓
billing/models → usage/storage → billing/services → payments/
```

## Security Architecture

### Authentication & Authorization
- **Path**: `api/authentication/`
- **Strategy**: JWT tokens with role-based permissions
- **Multi-tenancy**: Tenant-scoped data access

### Data Protection
- **Encryption**: At-rest (database) and in-transit (TLS)
- **PII Handling**: Dedicated models in `customers/models/`
- **Audit Trail**: All financial transactions logged

## Scalability Considerations

### Horizontal Scaling Points
1. **API Layer**: Stateless Django instances behind load balancer
2. **Task Processing**: Multiple Celery workers
3. **Database**: Read replicas for analytics queries
4. **Caching**: Redis cluster for session and query caching

### Performance Optimization
- **Database**: Optimized indexes in `db/indexes/`
- **Caching**: Multi-layer caching strategy in `cache/`
- **Async Processing**: Non-blocking operations via Celery

This architecture provides a solid foundation for a billing platform that can handle complex pricing models, high-volume usage data, and enterprise-scale operations while maintaining code clarity and system reliability.