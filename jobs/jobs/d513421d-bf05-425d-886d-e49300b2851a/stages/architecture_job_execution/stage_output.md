SPEC_ID: 2c83263f-4985-478a-a5ca-643500761929
SPEC_HASH: 29b0d0f3977574dd77639a494aa633f514e6d538aad0f3b97bff6de62d0a6965

# Orb Repository Architecture Map

## System Overview
Orb is a Python-based billing and usage metering platform built on FastAPI with PostgreSQL, designed for high-scale B2B SaaS billing operations.

## Core Architecture Layers

### 1. API Layer (`orb/`)
```
orb/
├── models/          # Pydantic models for API contracts
├── resources/       # FastAPI route handlers organized by domain
├── webhooks/        # Webhook handling and delivery
└── billing/         # Core billing engine
```

**Key Design Decision**: Domain-driven resource organization enables clear API boundaries and maintainable endpoint grouping.

### 2. Data Layer (`orb/models/`)
```
orb/models/
├── __init__.py                    # Model exports and registry
├── shared/                       # Cross-domain shared models
│   ├── __init__.py
│   ├── pagination.py            # Pagination contracts
│   └── discount.py              # Discount model definitions
├── customer.py                   # Customer domain models
├── subscription.py               # Subscription lifecycle models
├── plan.py                      # Pricing plan structures
├── invoice.py                   # Invoice and billing models
├── credit.py                    # Credit and prepayment models
├── usage.py                     # Usage metering models
├── price.py                     # Price configuration models
├── item.py                      # Billable item definitions
├── coupon.py                    # Coupon and promotion models
├── alert.py                     # Billing alert models
├── event.py                     # Usage event models
└── webhook.py                   # Webhook payload models
```

**Trade-off Analysis**: Separate model files per domain vs. consolidated models
- ✅ **Chosen**: Domain separation for maintainability
- ❌ **Rejected**: Single models file (would become unwieldy at scale)

### 3. Resource Handlers (`orb/resources/`)
```
orb/resources/
├── __init__.py
├── customers/                    # Customer management endpoints
│   ├── __init__.py
│   ├── customers.py             # CRUD operations
│   ├── costs.py                 # Cost analysis endpoints
│   ├── credits.py               # Credit management
│   └── usage.py                 # Usage querying
├── subscriptions/               # Subscription lifecycle
│   ├── __init__.py
│   ├── subscriptions.py         # Subscription CRUD
│   └── usage.py                 # Subscription usage
├── plans/                       # Plan management
│   ├── __init__.py
│   ├── plans.py                 # Plan CRUD operations
│   └── external_plan_id.py      # External ID mapping
├── invoices/                    # Invoice operations
│   ├── __init__.py
│   └── invoices.py              # Invoice generation/retrieval
├── items/                       # Billable items
│   ├── __init__.py
│   └── items.py                 # Item management
├── coupons/                     # Promotion management
│   ├── __init__.py
│   ├── coupons.py               # Coupon operations
│   └── subscriptions.py         # Coupon application
├── credits/                     # Credit operations
│   ├── __init__.py
│   └── credits.py               # Credit ledger management
├── prices/                      # Price configuration
│   ├── __init__.py
│   ├── prices.py                # Price CRUD
│   └── external_price_id.py     # External price mapping
├── events/                      # Usage event ingestion
│   ├── __init__.py
│   ├── events.py                # Event CRUD operations
│   ├── backfills.py             # Historical data backfill
│   └── volume.py                # Volume-based operations
├── alerts/                      # Billing alerts
│   ├── __init__.py
│   └── alerts.py                # Alert configuration
├── webhooks/                    # Webhook management
│   ├── __init__.py
│   └── webhooks.py              # Webhook CRUD
├── ping.py                      # Health check endpoint
└── top_level.py                 # Root-level operations
```

**Architectural Decision**: Nested resource structure mirrors API hierarchy
- **Benefit**: Intuitive code organization matching API design
- **Trade-off**: Deeper nesting vs. flat structure (chose clarity over simplicity)

### 4. Billing Engine (`orb/billing/`)
```
orb/billing/
├── __init__.py
└── (billing logic modules)
```

**Note**: Core billing computation and metering logic (implementation details would require deeper codebase analysis)

### 5. Webhook System (`orb/webhooks/`)
```
orb/webhooks/
├── __init__.py
└── (webhook delivery and retry logic)
```

**Design Consideration**: Separate webhook module ensures reliable event delivery with retry mechanisms

## Key Architectural Patterns

### 1. Domain-Driven Design (DDD)
- **Customer Domain**: Customer lifecycle, credits, usage analysis
- **Subscription Domain**: Plan associations, lifecycle management
- **Billing Domain**: Invoice generation, pricing calculations
- **Usage Domain**: Event ingestion, metering, backfills

### 2. Resource-Oriented Architecture (ROA)
Each business entity has dedicated resource handlers:
- Clear separation of concerns
- Consistent API patterns
- Maintainable endpoint organization

### 3. Model-View-Controller (MVC) Variant
- **Models**: Pydantic schemas in `orb/models/`
- **Controllers**: Resource handlers in `orb/resources/`
- **Views**: JSON API responses (implicit in FastAPI)

## Data Flow Architecture

```
External API Calls
        ↓
FastAPI Route Handlers (resources/)
        ↓
Pydantic Model Validation (models/)
        ↓
Business Logic Layer (billing/)
        ↓
Database Layer (PostgreSQL)
        ↓
Webhook Notifications (webhooks/)
```

## Scalability Considerations

### 1. Horizontal Scaling Points
- **API Layer**: Stateless FastAPI instances
- **Database**: PostgreSQL with read replicas
- **Webhook Delivery**: Async processing with retry queues

### 2. Performance Optimizations
- **Usage Events**: Bulk ingestion capabilities (`events/volume.py`)
- **Backfills**: Dedicated endpoints for historical data processing
- **Pagination**: Consistent pagination across all list endpoints

## Security Architecture

### 1. API Security
- FastAPI built-in validation and serialization
- Pydantic models provide input sanitization
- Resource-level authorization (implementation details in handlers)

### 2. Data Protection
- External ID mapping for customer data isolation
- Webhook signature verification (standard practice)

## Integration Points

### 1. External Systems
- **Customer Systems**: External plan/price ID mapping
- **Usage Sources**: Event ingestion endpoints
- **Notification Systems**: Webhook delivery

### 2. Internal Components
- **Billing Engine**: Core computation logic
- **Alert System**: Usage threshold monitoring
- **Credit System**: Prepayment and credit management

This architecture supports Orb's mission as a comprehensive billing platform while maintaining clear separation of concerns, scalability, and maintainability.