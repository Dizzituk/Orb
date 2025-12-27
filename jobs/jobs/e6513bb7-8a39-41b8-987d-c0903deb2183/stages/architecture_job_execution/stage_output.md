SPEC_ID: 015da9bb-7ced-47a1-8b15-b20221e635a9
SPEC_HASH: 2fdc4222f20d083acec0fd17a7459e04fdded7c547c321d7a70385d676a050ca

# Todo App REST API Architecture

## System Overview

A RESTful API for todo management with CRUD operations, pagination support, and extensible architecture for future tagging functionality.

## API Design

### Base URL Structure
```
https://api.todoapp.com/v1
```

### Core Endpoints

#### Todo CRUD Operations

```http
# Create todo
POST /todos
Content-Type: application/json
{
  "title": "Complete project documentation",
  "description": "Write comprehensive API docs",
  "priority": "high",
  "due_date": "2024-01-15T10:00:00Z",
  "status": "pending"
}

# Get todos with pagination
GET /todos?page=1&limit=20&sort=created_at&order=desc&status=pending

# Get specific todo
GET /todos/{id}

# Update todo
PUT /todos/{id}
Content-Type: application/json
{
  "title": "Updated title",
  "status": "completed"
}

# Partial update
PATCH /todos/{id}
Content-Type: application/json
{
  "status": "in_progress"
}

# Delete todo
DELETE /todos/{id}
```

### Response Formats

#### Single Todo Response
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Complete project documentation",
  "description": "Write comprehensive API docs",
  "status": "pending",
  "priority": "high",
  "due_date": "2024-01-15T10:00:00Z",
  "created_at": "2024-01-01T09:00:00Z",
  "updated_at": "2024-01-01T09:00:00Z",
  "tags": []  // Future extension point
}
```

#### Paginated Todos Response
```json
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Complete project documentation",
      // ... other fields
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 157,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  },
  "links": {
    "self": "/todos?page=1&limit=20",
    "next": "/todos?page=2&limit=20",
    "prev": null,
    "first": "/todos?page=1&limit=20",
    "last": "/todos?page=8&limit=20"
  }
}
```

## Data Model

### Todo Entity
```sql
CREATE TABLE todos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' 
        CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled')),
    priority VARCHAR(10) DEFAULT 'medium' 
        CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    due_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Future tagging support
    CONSTRAINT valid_title_length CHECK (LENGTH(title) > 0)
);

-- Indexes for performance
CREATE INDEX idx_todos_status ON todos(status);
CREATE INDEX idx_todos_due_date ON todos(due_date);
CREATE INDEX idx_todos_created_at ON todos(created_at);
CREATE INDEX idx_todos_priority ON todos(priority);
```

### Future Tag Support Schema
```sql
-- Prepared for future tagging feature
CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    color VARCHAR(7), -- hex color code
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE todo_tags (
    todo_id UUID REFERENCES todos(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (todo_id, tag_id)
);
```

## Technical Architecture

### Technology Stack
- **Runtime**: Node.js with TypeScript
- **Framework**: Express.js with Helmet, CORS
- **Database**: PostgreSQL with connection pooling
- **ORM**: Prisma or TypeORM
- **Validation**: Joi or Zod
- **Authentication**: JWT (future)
- **Rate Limiting**: Redis-backed rate limiter

### Project Structure
```
src/
├── controllers/
│   └── todoController.ts
├── services/
│   └── todoService.ts
├── repositories/
│   └── todoRepository.ts
├── models/
│   └── todo.ts
├── middleware/
│   ├── validation.ts
│   ├── pagination.ts
│   └── errorHandler.ts
├── routes/
│   └── todos.ts
├── utils/
│   ├── database.ts
│   └── logger.ts
└── app.ts
```

### Controller Implementation Example
```typescript
export class TodoController {
  constructor(private todoService: TodoService) {}

  async createTodo(req: Request, res: Response): Promise<void> {
    try {
      const todoData = await this.validateTodoInput(req.body);
      const todo = await this.todoService.createTodo(todoData);
      
      res.status(201).json({
        success: true,
        data: todo
      });
    } catch (error) {
      this.handleError(error, res);
    }
  }

  async getTodos(req: Request, res: Response): Promise<void> {
    try {
      const pagination = this.extractPaginationParams(req.query);
      const filters = this.extractFilters(req.query);
      
      const result = await this.todoService.getTodos(pagination, filters);
      
      res.json({
        success: true,
        ...result
      });
    } catch (error) {
      this.handleError(error, res);
    }
  }
}
```

## Key Design Decisions

### 1. UUID Primary Keys
- **Decision**: Use UUIDs instead of auto-incrementing integers
- **Rationale**: Better for distributed systems, no enumeration attacks, easier horizontal scaling
- **Trade-off**: Slightly larger storage footprint

### 2. Status Enumeration
- **Decision**: Database-level CHECK constraints for status values
- **Rationale**: Data integrity at the database level, prevents invalid states
- **Extension**: Easy to add new statuses via migration

### 3. Pagination Strategy
- **Decision**: Offset-based pagination with cursor option for future
- **Rationale**: Simple to implement, predictable for users
- **Future**: Can add cursor-based pagination for better performance at scale

### 4. Extensible Tag Architecture
- **Decision**: Separate tags table with many-to-many relationship
- **Rationale**: Normalized design, reusable tags, efficient queries
- **Performance**: Indexed for fast tag-based filtering

## Pagination Implementation

### Query Parameters
```typescript
interface PaginationParams {
  page?: number;      // Default: 1
  limit?: number;     // Default: 20, Max: 100
  sort?: string;      // Default: 'created_at'
  order?: 'asc' | 'desc'; // Default: 'desc'
}

interface FilterParams {
  status?: TodoStatus;
  priority?: TodoPriority;
  due_before?: string;
  due_after?: string;
  search?: string;    // Search in title/description
}
```

### Service Layer Logic
```typescript
async getTodos(
  pagination: PaginationParams,
  filters: FilterParams
): Promise<PaginatedResponse<Todo>> {
  const { page = 1, limit = 20, sort = 'created_at', order = 'desc' } = pagination;
  const offset = (page - 1) * limit;

  const whereClause = this.buildWhereClause(filters);
  const orderClause = { [sort]: order };

  const [todos, total] = await Promise.all([
    this.todoRepository.findMany({
      where: whereClause,
      orderBy: orderClause,
      take: limit,
      skip: offset
    }),
    this.todoRepository.count({ where: whereClause })
  ]);

  return this.buildPaginatedResponse(todos, total, page, limit);
}
```

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "title",
        "message": "Title is required"
      }
    ]
  },
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### HTTP Status Codes
- `200` - Success (GET, PUT, PATCH)
- `201` - Created (POST)
- `204` - No Content (DELETE)
- `400` - Bad Request (validation errors)
- `404` - Not Found
- `422` - Unprocessable Entity (business logic errors)
- `500` - Internal Server Error

## Security Considerations

### Input Validation
```typescript
const todoSchema = Joi.object({
  title: Joi.string().min(1).max(255).required(),
  description: Joi.string().max(2000).optional(),
  status: Joi.string().valid('pending', 'in_progress', 'completed', 'cancelled'),
  priority: Joi.string().valid('low', 'medium', 'high', 'urgent'),
  due_date: Joi.date().iso().optional()
});
```

### Rate Limiting
- 100 requests per minute per IP
- 1000 requests per hour per authenticated user (future)

### SQL Injection Prevention
- Parameterized queries via ORM
- Input sanitization and validation

## Performance Optimizations

### Database Indexes
- Composite indexes for common query patterns
- Partial indexes for filtered queries
- Consider covering indexes for read-heavy operations

### Caching Strategy (Future)
- Redis for frequently accessed todos
- Cache invalidation on updates
- ETags for conditional requests

### Connection Pooling
```typescript
const dbConfig = {
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME,
  username: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  pool: {
    min: 2,
    max: 10,
    acquire: 30000,
    idle: 10000
  }
};
```

## Future Extension Points

### 1. Tagging System
- Add tag endpoints: `GET /tags`, `POST /tags`, `DELETE /tags/{id}`
- Extend todo endpoints: `GET /todos?tags=work,urgent`
- Tag-based filtering and aggregation

### 2. User Management
- Multi-tenant architecture
- User-specific todos
- Sharing and collaboration features

### 3. Advanced Features
- Subtasks and todo hierarchies
- Recurring todos
- Attachments and notes
- Activity logging and audit trails

This architecture provides a solid foundation for a todo app with clear separation of concerns, extensibility for future features, and robust error handling and validation.