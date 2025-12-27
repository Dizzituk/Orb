SPEC_ID: 214c5db6-5bed-45ef-99fd-67f055fbca12
SPEC_HASH: 7c1dd74e4abb0b0f5cf9bef3a1a298d2a04efb9709f2b7abd324070bd1ef951d

# Orb Repository Architecture Map

## High-Level System Overview

Orb is a modern billing and usage-based pricing platform built with a microservices-oriented monolith architecture. The system follows domain-driven design principles with clear separation of concerns across billing, metering, pricing, and customer management domains.

## Core Architecture Layers

### 1. API Layer (`app/controllers/`)
```
app/controllers/
├── api/
│   └── v1/                           # Versioned API endpoints
│       ├── customers_controller.rb   # Customer management
│       ├── subscriptions_controller.rb # Subscription lifecycle
│       ├── invoices_controller.rb    # Invoice operations
│       ├── plans_controller.rb       # Pricing plan management
│       ├── events_controller.rb      # Usage event ingestion
│       └── webhooks_controller.rb    # Webhook endpoints
└── application_controller.rb         # Base controller with auth/middleware
```

### 2. Domain Models (`app/models/`)
```
app/models/
├── customer.rb                       # Customer entity and relationships
├── subscription.rb                   # Subscription lifecycle management
├── plan.rb                          # Pricing plan definitions
├── price.rb                         # Individual price components
├── invoice.rb                       # Invoice generation and management
├── invoice_line_item.rb             # Granular billing items
├── event.rb                         # Usage event model
├── credit.rb                        # Credit and prepaid balance
├── webhook_endpoint.rb              # Webhook configuration
└── concerns/
    ├── billable.rb                  # Shared billing behavior
    └── usage_trackable.rb           # Usage tracking mixins
```

### 3. Service Layer (`app/services/`)
```
app/services/
├── billing/
│   ├── invoice_generator.rb         # Core invoice creation logic
│   ├── proration_calculator.rb      # Subscription change calculations
│   └── credit_applicator.rb         # Credit application logic
├── pricing/
│   ├── plan_calculator.rb           # Plan-based pricing calculations
│   ├── usage_aggregator.rb          # Usage event aggregation
│   └── tier_calculator.rb           # Tiered pricing logic
├── events/
│   ├── ingestion_service.rb         # Event validation and storage
│   └── deduplication_service.rb     # Event deduplication
└── webhooks/
    ├── delivery_service.rb          # Webhook delivery management
    └── retry_service.rb             # Failed webhook retry logic
```

### 4. Background Jobs (`app/jobs/`)
```
app/jobs/
├── billing/
│   ├── invoice_generation_job.rb    # Scheduled invoice creation
│   ├── payment_retry_job.rb         # Failed payment retry
│   └── subscription_renewal_job.rb  # Subscription renewal processing
├── events/
│   ├── event_processing_job.rb      # Async event processing
│   └── usage_aggregation_job.rb     # Periodic usage aggregation
└── webhooks/
    ├── webhook_delivery_job.rb      # Async webhook delivery
    └── webhook_retry_job.rb         # Webhook retry processing
```

### 5. Database Layer
```
db/
├── migrate/                         # Database migrations
│   ├── 001_create_customers.rb
│   ├── 002_create_plans.rb
│   ├── 003_create_subscriptions.rb
│   ├── 004_create_invoices.rb
│   ├── 005_create_events.rb
│   └── ...
└── schema.rb                        # Current database schema
```

### 6. Configuration & Infrastructure
```
config/
├── application.rb                   # Rails application configuration
├── database.yml                     # Database configuration
├── routes.rb                        # API route definitions
├── initializers/
│   ├── redis.rb                     # Redis configuration for caching/jobs
│   ├── sidekiq.rb                   # Background job processing
│   └── cors.rb                      # CORS configuration
└── environments/
    ├── development.rb
    ├── test.rb
    └── production.rb
```

### 7. Testing Infrastructure
```
spec/
├── models/                          # Model unit tests
├── controllers/                     # Controller integration tests
├── services/                        # Service layer tests
├── jobs/                           # Background job tests
├── factories/                       # Test data factories
└── support/
    ├── database_cleaner.rb
    └── shared_examples/
```

## Key Architectural Patterns

### 1. Domain-Driven Design
- **Customer Domain**: Customer management, authentication, organization
- **Billing Domain**: Invoicing, payments, credits, taxation
- **Pricing Domain**: Plans, prices, tiers, usage calculations
- **Events Domain**: Usage tracking, event ingestion, aggregation

### 2. Service Object Pattern
Each complex business operation is encapsulated in dedicated service objects:
- `BillingService::InvoiceGenerator` - Handles invoice creation logic
- `PricingService::UsageAggregator` - Aggregates usage events for billing
- `EventService::IngestionService` - Validates and processes incoming events

### 3. Background Job Architecture
- **Sidekiq** for job processing with Redis backing
- **Scheduled jobs** for recurring billing operations
- **Event-driven jobs** for webhook delivery and retries
- **Priority queues** for time-sensitive operations

### 4. API Design Patterns
- **RESTful endpoints** with consistent resource naming
- **API versioning** through URL path (`/api/v1/`)
- **Standardized error responses** with proper HTTP status codes
- **Pagination** for list endpoints
- **Filtering and sorting** capabilities

## Data Flow Architecture

### Usage Event Ingestion Flow
```
External System → API Controller → Validation → Event Model → Background Job → Aggregation → Billing
```

### Invoice Generation Flow
```
Scheduler → Job → Service → Usage Calculation → Invoice Creation → Webhook Delivery
```

### Subscription Management Flow
```
API Request → Controller → Service → Model Updates → Event Triggers → Webhook Notifications
```

## Key Technical Decisions

### 1. Database Design
- **PostgreSQL** as primary database for ACID compliance and complex queries
- **Optimized indexes** on frequently queried fields (customer_id, subscription_id, created_at)
- **Partitioning strategy** for events table by time periods
- **Foreign key constraints** to maintain referential integrity

### 2. Caching Strategy
- **Redis** for session storage and job queues
- **Application-level caching** for frequently accessed pricing data
- **Database query caching** for expensive aggregation queries

### 3. API Rate Limiting
- **Per-customer rate limiting** to prevent abuse
- **Endpoint-specific limits** based on operation cost
- **Graceful degradation** with proper error responses

### 4. Security Architecture
- **JWT-based authentication** for API access
- **Role-based access control** (RBAC) for different user types
- **API key management** for programmatic access
- **Input validation** at controller and service levels
- **SQL injection prevention** through parameterized queries

## Scalability Considerations

### 1. Horizontal Scaling Points
- **API servers** can be load-balanced across multiple instances
- **Background job workers** can be scaled independently
- **Database read replicas** for read-heavy operations

### 2. Performance Optimizations
- **Eager loading** to prevent N+1 queries
- **Bulk operations** for large data processing
- **Async processing** for non-critical operations
- **Connection pooling** for database efficiency

### 3. Monitoring & Observability
- **Application metrics** tracking key business KPIs
- **Performance monitoring** for API response times
- **Error tracking** with detailed stack traces
- **Database query analysis** for optimization opportunities

This architecture provides a solid foundation for a billing platform with clear separation of concerns, scalability considerations, and maintainable code organization.