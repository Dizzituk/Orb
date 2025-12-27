SPEC_ID: 9ef932bb-a911-42d6-ad28-b2be09c0ec48
SPEC_HASH: eec82822d81e9d35702868be54dc84e027eacbb6f9b9d648a620dd31bf185998

# Real-Time Usage Monitoring & Alerting System

## System Overview

I'll design a real-time usage monitoring and alerting system for Orb's billing platform that tracks customer usage patterns, detects anomalies, and triggers automated alerts for billing events.

## Architecture Design

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Usage Ingress │    │  Stream Processor│    │  Alert Engine   │
│                 │───▶│                 │───▶│                 │
│ - Kafka Topics  │    │ - Flink/Kafka   │    │ - Rule Engine   │
│ - Rate Limiting │    │ - Aggregations  │    │ - Notifications │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Storage   │    │  Metrics Store  │    │  Alert Storage  │
│                 │    │                 │    │                 │
│ - S3/Parquet    │    │ - ClickHouse    │    │ - PostgreSQL    │
│ - Partitioned   │    │ - Time Series   │    │ - Audit Trail   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow Architecture

```python
# Usage Event Schema
@dataclass
class UsageEvent:
    customer_id: str
    metric_name: str
    value: float
    timestamp: datetime
    dimensions: Dict[str, str]
    source: str
    event_id: str
    
# Processed Metrics Schema
@dataclass  
class ProcessedMetric:
    customer_id: str
    metric_name: str
    aggregated_value: float
    time_window: str
    dimensions: Dict[str, str]
    processing_timestamp: datetime
```

## Implementation Details

### 1. Usage Ingress Layer

```python
class UsageIngressService:
    def __init__(self):
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['kafka-1:9092', 'kafka-2:9092'],
            value_serializer=lambda v: json.dumps(v).encode(),
            compression_type='snappy',
            batch_size=16384,
            linger_ms=10
        )
        self.rate_limiter = TokenBucketRateLimiter()
    
    async def ingest_usage(self, events: List[UsageEvent]) -> IngestResponse:
        # Validate and enrich events
        validated_events = []
        for event in events:
            if not self._validate_event(event):
                continue
            
            # Add processing metadata
            enriched_event = {
                **event.__dict__,
                'ingress_timestamp': datetime.utcnow().isoformat(),
                'partition_key': f"{event.customer_id}:{event.metric_name}"
            }
            validated_events.append(enriched_event)
        
        # Rate limiting per customer
        for event in validated_events:
            if not await self.rate_limiter.allow(event['customer_id']):
                raise RateLimitExceedException(event['customer_id'])
        
        # Batch send to Kafka
        futures = []
        for event in validated_events:
            future = self.kafka_producer.send(
                topic='usage-events',
                key=event['partition_key'].encode(),
                value=event
            )
            futures.append(future)
        
        # Wait for acknowledgments
        await asyncio.gather(*futures)
        
        return IngestResponse(
            accepted_count=len(validated_events),
            rejected_count=len(events) - len(validated_events)
        )
```

### 2. Stream Processing Engine

```python
class UsageStreamProcessor:
    def __init__(self):
        self.flink_env = StreamExecutionEnvironment.get_execution_environment()
        self.flink_env.set_parallelism(8)
        self.flink_env.enable_checkpointing(30000)  # 30 second checkpoints
    
    def create_processing_pipeline(self):
        # Source: Kafka usage events
        kafka_source = FlinkKafkaConsumer(
            topics=['usage-events'],
            deserialization_schema=JSONDeserializationSchema(),
            properties={
                'bootstrap.servers': 'kafka-1:9092,kafka-2:9092',
                'group.id': 'usage-processor'
            }
        )
        
        usage_stream = self.flink_env.add_source(kafka_source)
        
        # Windowed aggregations
        aggregated_metrics = (usage_stream
            .key_by(lambda event: f"{event['customer_id']}:{event['metric_name']}")
            .time_window(Time.minutes(5))  # 5-minute windows
            .aggregate(UsageAggregator())
            .name("usage-aggregation"))
        
        # Anomaly detection
        anomaly_stream = (aggregated_metrics
            .key_by(lambda metric: metric.customer_id)
            .process(AnomalyDetectionFunction())
            .name("anomaly-detection"))
        
        # Sinks
        aggregated_metrics.add_sink(ClickHouseSink())
        anomaly_stream.add_sink(AlertKafkaSink())
        
        return self.flink_env

class UsageAggregator(AggregateFunction):
    def create_accumulator(self) -> Dict:
        return {
            'sum': 0.0,
            'count': 0,
            'min': float('inf'),
            'max': float('-inf'),
            'customer_id': None,
            'metric_name': None
        }
    
    def add(self, event: Dict, accumulator: Dict) -> Dict:
        accumulator['sum'] += event['value']
        accumulator['count'] += 1
        accumulator['min'] = min(accumulator['min'], event['value'])
        accumulator['max'] = max(accumulator['max'], event['value'])
        accumulator['customer_id'] = event['customer_id']
        accumulator['metric_name'] = event['metric_name']
        return accumulator
    
    def get_result(self, accumulator: Dict) -> ProcessedMetric:
        return ProcessedMetric(
            customer_id=accumulator['customer_id'],
            metric_name=accumulator['metric_name'],
            aggregated_value=accumulator['sum'],
            time_window='5m',
            dimensions={
                'count': accumulator['count'],
                'avg': accumulator['sum'] / accumulator['count'],
                'min': accumulator['min'],
                'max': accumulator['max']
            },
            processing_timestamp=datetime.utcnow()
        )
```

### 3. Alert Engine

```python
class AlertEngine:
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.notification_service = NotificationService()
        self.alert_store = AlertStore()
    
    async def process_alert_triggers(self, metric: ProcessedMetric):
        # Evaluate alert rules
        triggered_rules = await self.rule_engine.evaluate(metric)
        
        for rule in triggered_rules:
            alert = Alert(
                id=str(uuid.uuid4()),
                customer_id=metric.customer_id,
                rule_id=rule.id,
                metric_name=metric.metric_name,
                threshold_value=rule.threshold,
                actual_value=metric.aggregated_value,
                severity=rule.severity,
                created_at=datetime.utcnow(),
                status=AlertStatus.ACTIVE
            )
            
            # Store alert
            await self.alert_store.save(alert)
            
            # Send notifications
            await self._send_notifications(alert, rule)
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        notification_channels = rule.notification_channels
        
        tasks = []
        for channel in notification_channels:
            if channel.type == 'webhook':
                tasks.append(self._send_webhook(alert, channel))
            elif channel.type == 'email':
                tasks.append(self._send_email(alert, channel))
            elif channel.type == 'slack':
                tasks.append(self._send_slack(alert, channel))
        
        await asyncio.gather(*tasks, return_exceptions=True)

class RuleEngine:
    def __init__(self):
        self.rules_cache = TTLCache(maxsize=10000, ttl=300)  # 5 min cache
    
    async def evaluate(self, metric: ProcessedMetric) -> List[AlertRule]:
        # Get rules for customer and metric
        cache_key = f"{metric.customer_id}:{metric.metric_name}"
        rules = self.rules_cache.get(cache_key)
        
        if rules is None:
            rules = await self._fetch_rules(metric.customer_id, metric.metric_name)
            self.rules_cache[cache_key] = rules
        
        triggered_rules = []
        for rule in rules:
            if self._evaluate_condition(metric, rule):
                triggered_rules.append(rule)
        
        return triggered_rules
    
    def _evaluate_condition(self, metric: ProcessedMetric, rule: AlertRule) -> bool:
        value = metric.aggregated_value
        
        if rule.condition_type == 'threshold':
            return self._check_threshold(value, rule.threshold, rule.operator)
        elif rule.condition_type == 'anomaly':
            return self._check_anomaly(metric, rule)
        elif rule.condition_type == 'rate_change':
            return self._check_rate_change(metric, rule)
        
        return False
```

### 4. Data Storage Layer

```sql
-- ClickHouse schema for metrics storage
CREATE TABLE usage_metrics (
    customer_id String,
    metric_name String,
    aggregated_value Float64,
    time_window String,
    dimensions Map(String, String),
    processing_timestamp DateTime64(3),
    date Date DEFAULT toDate(processing_timestamp)
) ENGINE = MergeTree()
PARTITION BY date
ORDER BY (customer_id, metric_name, processing_timestamp)
TTL processing_timestamp + INTERVAL 90 DAY;

-- PostgreSQL schema for alerts
CREATE TABLE alerts (
    id UUID PRIMARY KEY,
    customer_id VARCHAR(255) NOT NULL,
    rule_id UUID NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    threshold_value DECIMAL(20,8),
    actual_value DECIMAL(20,8),
    severity VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

CREATE INDEX idx_alerts_customer_status ON alerts(customer_id, status);
CREATE INDEX idx_alerts_created_at ON alerts(created_at);
```

## Key Design Decisions

### 1. Stream Processing Choice
**Decision**: Apache Flink over Apache Spark
**Rationale**: 
- Lower latency for real-time processing
- Better backpressure handling
- Native support for event time processing
- Exactly-once processing guarantees

### 2. Storage Strategy
**Decision**: Hybrid storage (ClickHouse + PostgreSQL)
**Rationale**:
- ClickHouse for high-volume time-series metrics (fast aggregations)
- PostgreSQL for transactional alert data (ACID compliance)
- Separate concerns for different access patterns

### 3. Partitioning Strategy
**Decision**: Partition by customer_id + metric_name
**Rationale**:
- Enables parallel processing per customer
- Reduces hot partitions
- Aligns with query patterns

## Scalability Considerations

### Horizontal Scaling
```yaml
# Kubernetes deployment for stream processor
apiVersion: apps/v1
kind: Deployment
metadata:
  name: usage-stream-processor
spec:
  replicas: 8
  selector:
    matchLabels:
      app: usage-stream-processor
  template:
    spec:
      containers:
      - name: flink-taskmanager
        image: flink:1.17
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: FLINK_PARALLELISM
          value: "4"
```

### Performance Optimizations
- Kafka topic partitioning: 32 partitions per topic
- Flink parallelism: 8 task managers, 4 slots each
- ClickHouse sharding: 4 shards across availability zones
- Connection pooling: 100 connections per service

## Security & Compliance

### Data Protection
```python
class DataEncryption:
    def __init__(self):
        self.encryption_key = self._load_encryption_key()
    
    def encrypt_sensitive_data(self, data: Dict) -> Dict:
        sensitive_fields = ['customer_id', 'dimensions']
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in data:
                encrypted_data[field] = self._encrypt(str(data[field]))
        
        return encrypted_data
    
    def _encrypt(self, plaintext: str) -> str:
        cipher = Fernet(self.encryption_key)
        return cipher.encrypt(plaintext.encode()).decode()
```

### Access Control
- API authentication via JWT tokens
- Role-based access control (RBAC)
- Customer data isolation
- Audit logging for all operations

## Monitoring & Observability

### Key Metrics
- Usage ingestion rate (events/second)
- Stream processing latency (p95, p99)
- Alert generation rate
- Data loss percentage
- System resource utilization

### Health Checks
```python
class SystemHealthCheck:
    async def check_system_health(self) -> HealthStatus:
        checks = await asyncio.gather(
            self._check_kafka_connectivity(),
            self._check_flink_job_status(),
            self._check_clickhouse_availability(),
            self._check_alert_engine_status()
        )
        
        return HealthStatus(
            overall_status='healthy' if all(checks) else 'degraded',
            component_status={
                'kafka': checks[0],
                'flink': checks[1], 
                'clickhouse': checks[2],
                'alerts': checks[3]
            }
        )
```

This architecture provides a robust, scalable real-time usage monitoring system that can handle high-volume usage data while maintaining low latency for critical billing alerts.