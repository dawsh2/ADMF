# ADMF-Trader: Production Deployment & Operations

This document covers production-ready deployment, monitoring, error recovery, performance tuning, and security for the ADMF-Trader system.

## Table of Contents

1. [Deployment Patterns](#1-deployment-patterns)
2. [Monitoring & Alerting](#2-monitoring--alerting)
3. [Error Recovery Strategies](#3-error-recovery-strategies)
4. [Performance Tuning](#4-performance-tuning)
5. [Security Considerations](#5-security-considerations)

---

## 1. Deployment Patterns

### 1.1 Deployment Architectures

```python
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import docker
import kubernetes
from abc import abstractmethod

class DeploymentMode(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"

class ScalingStrategy(Enum):
    SINGLE_INSTANCE = "single_instance"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    AUTO_SCALING = "auto_scaling"

@dataclass
class DeploymentConfig:
    """Configuration for deployment"""
    mode: DeploymentMode
    scaling_strategy: ScalingStrategy
    resource_limits: Dict[str, Any]
    health_check_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    backup_config: Dict[str, Any]
    security_config: Dict[str, Any]

class DeploymentManager:
    """Manages system deployment across environments"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_history: List[Dict[str, Any]] = []
        self.current_deployment: Optional[Dict[str, Any]] = None
    
    def deploy(self, version: str, environment: str) -> bool:
        """Deploy system to specified environment"""
        deployment_record = {
            'version': version,
            'environment': environment,
            'timestamp': datetime.now(),
            'config': self.config,
            'status': 'in_progress'
        }
        
        try:
            # Pre-deployment validation
            if not self._validate_deployment_readiness():
                raise DeploymentError("System not ready for deployment")
            
            # Environment-specific deployment
            if environment == "production":
                success = self._deploy_production(version)
            elif environment == "staging":
                success = self._deploy_staging(version)
            elif environment == "development":
                success = self._deploy_development(version)
            else:
                raise ValueError(f"Unknown environment: {environment}")
            
            deployment_record['status'] = 'success' if success else 'failed'
            self.deployment_history.append(deployment_record)
            
            if success:
                self.current_deployment = deployment_record
            
            return success
            
        except Exception as e:
            deployment_record['status'] = 'failed'
            deployment_record['error'] = str(e)
            self.deployment_history.append(deployment_record)
            return False
    
    def _validate_deployment_readiness(self) -> bool:
        """Validate system is ready for deployment"""
        checks = [
            self._check_configuration_validity(),
            self._check_dependencies_available(),
            self._check_database_connectivity(),
            self._check_external_services(),
            self._check_resource_availability()
        ]
        
        return all(checks)
    
    def _deploy_production(self, version: str) -> bool:
        """Deploy to production environment"""
        # Blue-Green deployment for zero downtime
        return self._blue_green_deployment(version)
    
    def _deploy_staging(self, version: str) -> bool:
        """Deploy to staging environment"""
        # Rolling deployment for staging
        return self._rolling_deployment(version)
    
    def _deploy_development(self, version: str) -> bool:
        """Deploy to development environment"""
        # Simple deployment for development
        return self._simple_deployment(version)
    
    def _blue_green_deployment(self, version: str) -> bool:
        """Blue-Green deployment pattern"""
        try:
            # 1. Deploy to green environment
            green_env = self._create_green_environment(version)
            
            # 2. Run health checks on green
            if not self._verify_green_environment(green_env):
                self._cleanup_green_environment(green_env)
                return False
            
            # 3. Switch traffic to green
            self._switch_traffic_to_green(green_env)
            
            # 4. Monitor for issues
            if self._monitor_deployment_health(duration=300):  # 5 minutes
                # 5. Cleanup old blue environment
                self._cleanup_blue_environment()
                return True
            else:
                # Rollback to blue
                self._rollback_to_blue()
                return False
                
        except Exception as e:
            print(f"Blue-green deployment failed: {e}")
            return False
    
    def _rolling_deployment(self, version: str) -> bool:
        """Rolling deployment pattern"""
        try:
            instances = self._get_current_instances()
            
            for instance in instances:
                # Update instance one at a time
                self._update_instance(instance, version)
                
                # Health check after each update
                if not self._health_check_instance(instance):
                    # Rollback this instance
                    self._rollback_instance(instance)
                    return False
                
                # Wait between updates
                time.sleep(30)
            
            return True
            
        except Exception as e:
            print(f"Rolling deployment failed: {e}")
            return False

class ContainerDeployment:
    """Docker container deployment management"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.containers: Dict[str, Any] = {}
    
    def build_image(self, dockerfile_path: str, tag: str) -> bool:
        """Build Docker image"""
        try:
            image, logs = self.docker_client.images.build(
                path=dockerfile_path,
                tag=tag,
                rm=True
            )
            return True
        except Exception as e:
            print(f"Failed to build image: {e}")
            return False
    
    def deploy_container(self, image_tag: str, container_name: str, 
                        config: Dict[str, Any]) -> bool:
        """Deploy container"""
        try:
            # Stop existing container if running
            self._stop_container(container_name)
            
            # Start new container
            container = self.docker_client.containers.run(
                image=image_tag,
                name=container_name,
                environment=config.get('environment', {}),
                ports=config.get('ports', {}),
                volumes=config.get('volumes', {}),
                restart_policy=config.get('restart_policy', {'Name': 'always'}),
                detach=True
            )
            
            self.containers[container_name] = container
            
            # Health check
            return self._container_health_check(container_name)
            
        except Exception as e:
            print(f"Failed to deploy container {container_name}: {e}")
            return False
    
    def _container_health_check(self, container_name: str) -> bool:
        """Check container health"""
        try:
            container = self.containers[container_name]
            
            # Wait for container to start
            time.sleep(10)
            
            # Check if container is running
            container.reload()
            if container.status != 'running':
                return False
            
            # Custom health check via HTTP endpoint
            import requests
            health_url = f"http://localhost:8080/health"  # Configurable
            response = requests.get(health_url, timeout=30)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Container health check failed: {e}")
            return False

class KubernetesDeployment:
    """Kubernetes deployment management"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        from kubernetes import client, config
        
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_incluster_config()
        
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.networking_v1 = client.NetworkingV1Api()
    
    def deploy_application(self, namespace: str, deployment_spec: Dict[str, Any]) -> bool:
        """Deploy application to Kubernetes"""
        try:
            # Create deployment
            deployment = self._create_deployment(namespace, deployment_spec)
            
            # Create service
            service = self._create_service(namespace, deployment_spec)
            
            # Create ingress if specified
            if deployment_spec.get('ingress'):
                ingress = self._create_ingress(namespace, deployment_spec)
            
            # Wait for deployment to be ready
            return self._wait_for_deployment_ready(namespace, deployment.metadata.name)
            
        except Exception as e:
            print(f"Kubernetes deployment failed: {e}")
            return False
    
    def _create_deployment(self, namespace: str, spec: Dict[str, Any]):
        """Create Kubernetes deployment"""
        from kubernetes import client
        
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=spec['name'],
                labels=spec.get('labels', {})
            ),
            spec=client.V1DeploymentSpec(
                replicas=spec.get('replicas', 1),
                selector=client.V1LabelSelector(
                    match_labels=spec.get('selector_labels', {})
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels=spec.get('pod_labels', {})
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=container['name'],
                                image=container['image'],
                                ports=[client.V1ContainerPort(container_port=port) 
                                       for port in container.get('ports', [])],
                                env=[client.V1EnvVar(name=k, value=v) 
                                     for k, v in container.get('environment', {}).items()],
                                resources=client.V1ResourceRequirements(
                                    requests=container.get('requests', {}),
                                    limits=container.get('limits', {})
                                ),
                                liveness_probe=self._create_probe(container.get('liveness_probe')),
                                readiness_probe=self._create_probe(container.get('readiness_probe'))
                            ) for container in spec['containers']
                        ]
                    )
                )
            )
        )
        
        return self.apps_v1.create_namespaced_deployment(
            namespace=namespace,
            body=deployment
        )

class DeploymentOrchestrator:
    """Orchestrates complex multi-component deployments"""
    
    def __init__(self):
        self.deployment_graph = nx.DiGraph()
        self.deployment_status: Dict[str, str] = {}
    
    def add_component(self, component_name: str, dependencies: List[str] = None):
        """Add component to deployment graph"""
        self.deployment_graph.add_node(component_name)
        
        if dependencies:
            for dep in dependencies:
                self.deployment_graph.add_edge(dep, component_name)
    
    def deploy_system(self, environment: str) -> bool:
        """Deploy entire system in dependency order"""
        try:
            # Get deployment order
            deployment_order = list(nx.topological_sort(self.deployment_graph))
            
            for component in deployment_order:
                success = self._deploy_component(component, environment)
                self.deployment_status[component] = 'success' if success else 'failed'
                
                if not success:
                    # Rollback deployed components
                    self._rollback_deployment(deployment_order[:deployment_order.index(component)])
                    return False
            
            return True
            
        except Exception as e:
            print(f"System deployment failed: {e}")
            return False
    
    def _deploy_component(self, component: str, environment: str) -> bool:
        """Deploy individual component"""
        # Component-specific deployment logic
        # This would integrate with ContainerDeployment or KubernetesDeployment
        pass
```

### 1.2 Infrastructure as Code

```yaml
# docker-compose.yml for development
version: '3.8'
services:
  admf-trader:
    build: 
      context: .
      dockerfile: Dockerfile
    container_name: admf-trader-dev
    environment:
      - ADMF_ENV=development
      - ADMF_LOG_LEVEL=DEBUG
      - ADMF_DB_URL=postgresql://user:pass@db:5432/admf_dev
    ports:
      - "8080:8080"
      - "8081:8081"  # Health check port
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgresql:13
    environment:
      - POSTGRES_DB=admf_dev
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

```yaml
# kubernetes/deployment.yaml for production
apiVersion: apps/v1
kind: Deployment
metadata:
  name: admf-trader
  namespace: trading
  labels:
    app: admf-trader
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: admf-trader
  template:
    metadata:
      labels:
        app: admf-trader
        version: v1.0.0
    spec:
      containers:
      - name: admf-trader
        image: admf-trader:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: health
        env:
        - name: ADMF_ENV
          value: "production"
        - name: ADMF_DB_URL
          valueFrom:
            secretKeyRef:
              name: admf-secrets
              key: database-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: config-volume
        configMap:
          name: admf-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: admf-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: admf-trader-service
  namespace: trading
spec:
  selector:
    app: admf-trader
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: health
    port: 8081
    targetPort: 8081
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: admf-trader-ingress
  namespace: trading
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - trading.example.com
    secretName: admf-tls
  rules:
  - host: trading.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: admf-trader-service
            port:
              number: 80
```

---

## 2. Monitoring & Alerting

### 2.1 Comprehensive Monitoring System

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
import threading
from typing import Dict, Any, List
from dataclasses import dataclass

class MetricsCollector:
    """Production metrics collection"""
    
    def __init__(self):
        # Trading metrics
        self.trades_total = Counter('admf_trades_total', 'Total number of trades executed', ['strategy', 'symbol', 'direction'])
        self.trade_pnl = Histogram('admf_trade_pnl', 'Trade P&L distribution', ['strategy'])
        self.portfolio_value = Gauge('admf_portfolio_value', 'Current portfolio value')
        self.drawdown = Gauge('admf_drawdown_percent', 'Current drawdown percentage')
        self.signals_generated = Counter('admf_signals_generated_total', 'Total signals generated', ['strategy', 'symbol'])
        self.signals_filtered = Counter('admf_signals_filtered_total', 'Total signals filtered', ['reason'])
        
        # System metrics
        self.component_status = Gauge('admf_component_status', 'Component health status', ['component'])
        self.event_processing_time = Histogram('admf_event_processing_seconds', 'Event processing time', ['event_type'])
        self.cache_hits = Counter('admf_cache_hits_total', 'Cache hits', ['cache_name'])
        self.cache_misses = Counter('admf_cache_misses_total', 'Cache misses', ['cache_name'])
        self.errors_total = Counter('admf_errors_total', 'Total errors', ['component', 'error_type'])
        
        # Performance metrics
        self.cpu_usage = Gauge('admf_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('admf_memory_usage_bytes', 'Memory usage in bytes')
        self.disk_usage = Gauge('admf_disk_usage_percent', 'Disk usage percentage')
        self.network_io = Counter('admf_network_io_bytes_total', 'Network I/O', ['direction'])
        
        # Data quality metrics
        self.data_latency = Histogram('admf_data_latency_seconds', 'Data latency', ['source'])
        self.missing_data_points = Counter('admf_missing_data_points_total', 'Missing data points', ['symbol'])
        self.data_quality_score = Gauge('admf_data_quality_score', 'Data quality score', ['symbol'])
    
    def record_trade(self, strategy: str, symbol: str, direction: str, pnl: float):
        """Record trade metrics"""
        self.trades_total.labels(strategy=strategy, symbol=symbol, direction=direction).inc()
        self.trade_pnl.labels(strategy=strategy).observe(pnl)
    
    def update_portfolio_metrics(self, value: float, drawdown_pct: float):
        """Update portfolio metrics"""
        self.portfolio_value.set(value)
        self.drawdown.set(drawdown_pct)
    
    def record_signal(self, strategy: str, symbol: str, filtered: bool = False, filter_reason: str = None):
        """Record signal metrics"""
        if filtered:
            self.signals_filtered.labels(reason=filter_reason or 'unknown').inc()
        else:
            self.signals_generated.labels(strategy=strategy, symbol=symbol).inc()
    
    def record_event_processing(self, event_type: str, processing_time: float):
        """Record event processing metrics"""
        self.event_processing_time.labels(event_type=event_type).observe(processing_time)
    
    def update_component_status(self, component: str, healthy: bool):
        """Update component health status"""
        self.component_status.labels(component=component).set(1 if healthy else 0)
    
    def record_cache_access(self, cache_name: str, hit: bool):
        """Record cache access"""
        if hit:
            self.cache_hits.labels(cache_name=cache_name).inc()
        else:
            self.cache_misses.labels(cache_name=cache_name).inc()
    
    def record_error(self, component: str, error_type: str):
        """Record error occurrence"""
        self.errors_total.labels(component=component, error_type=error_type).inc()

class AlertManager:
    """Production alerting system"""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.notification_channels: List[Any] = []
        self.alert_suppression: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def add_alert_rule(self, name: str, condition: Callable, severity: str, 
                      notification_channels: List[str], suppression_minutes: int = 5):
        """Add alert rule"""
        self.alert_rules[name] = {
            'condition': condition,
            'severity': severity,
            'notification_channels': notification_channels,
            'suppression_minutes': suppression_minutes,
            'active': False
        }
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert conditions"""
        with self._lock:
            for rule_name, rule in self.alert_rules.items():
                try:
                    # Check if alert is suppressed
                    if self._is_alert_suppressed(rule_name):
                        continue
                    
                    # Evaluate condition
                    triggered = rule['condition'](metrics)
                    
                    if triggered and not rule['active']:
                        # New alert
                        self._trigger_alert(rule_name, rule, metrics)
                        rule['active'] = True
                    elif not triggered and rule['active']:
                        # Alert resolved
                        self._resolve_alert(rule_name, rule)
                        rule['active'] = False
                        
                except Exception as e:
                    print(f"Error checking alert rule {rule_name}: {e}")
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger an alert"""
        alert = {
            'rule_name': rule_name,
            'severity': rule['severity'],
            'timestamp': datetime.now(),
            'status': 'triggered',
            'metrics': metrics.copy(),
            'message': self._generate_alert_message(rule_name, metrics)
        }
        
        self.alert_history.append(alert)
        
        # Send notifications
        for channel_name in rule['notification_channels']:
            self._send_notification(channel_name, alert)
        
        # Set suppression
        suppression_until = datetime.now() + timedelta(minutes=rule['suppression_minutes'])
        self.alert_suppression[rule_name] = suppression_until
    
    def _resolve_alert(self, rule_name: str, rule: Dict[str, Any]):
        """Resolve an alert"""
        alert = {
            'rule_name': rule_name,
            'severity': rule['severity'],
            'timestamp': datetime.now(),
            'status': 'resolved',
            'message': f"Alert {rule_name} has been resolved"
        }
        
        self.alert_history.append(alert)
        
        # Send resolution notifications
        for channel_name in rule['notification_channels']:
            self._send_notification(channel_name, alert)
    
    def _is_alert_suppressed(self, rule_name: str) -> bool:
        """Check if alert is currently suppressed"""
        if rule_name in self.alert_suppression:
            return datetime.now() < self.alert_suppression[rule_name]
        return False
    
    def add_notification_channel(self, channel: Any):
        """Add notification channel"""
        self.notification_channels.append(channel)

class EmailNotificationChannel:
    """Email notification channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients: List[str] = []
    
    def add_recipient(self, email: str):
        """Add email recipient"""
        self.recipients.append(email)
    
    def send_notification(self, alert: Dict[str, Any]):
        """Send email notification"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['Subject'] = f"ADMF Alert: {alert['rule_name']} - {alert['severity'].upper()}"
            
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            
            for recipient in self.recipients:
                msg['To'] = recipient
                server.send_message(msg)
                del msg['To']
            
            server.quit()
            
        except Exception as e:
            print(f"Failed to send email notification: {e}")
    
    def _format_email_body(self, alert: Dict[str, Any]) -> str:
        """Format email body"""
        severity_color = {
            'critical': '#ff0000',
            'warning': '#ffaa00',
            'info': '#0066cc'
        }.get(alert['severity'], '#666666')
        
        return f"""
        <html>
        <body>
            <h2 style="color: {severity_color};">ADMF Trading System Alert</h2>
            <p><strong>Rule:</strong> {alert['rule_name']}</p>
            <p><strong>Severity:</strong> {alert['severity'].upper()}</p>
            <p><strong>Status:</strong> {alert['status'].upper()}</p>
            <p><strong>Time:</strong> {alert['timestamp']}</p>
            <p><strong>Message:</strong> {alert['message']}</p>
            
            {self._format_metrics_table(alert.get('metrics', {}))}
        </body>
        </html>
        """

class SlackNotificationChannel:
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_notification(self, alert: Dict[str, Any]):
        """Send Slack notification"""
        import requests
        
        color = {
            'critical': 'danger',
            'warning': 'warning',
            'info': 'good'
        }.get(alert['severity'], 'warning')
        
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"ADMF Alert: {alert['rule_name']}",
                    "fields": [
                        {"title": "Severity", "value": alert['severity'].upper(), "short": True},
                        {"title": "Status", "value": alert['status'].upper(), "short": True},
                        {"title": "Time", "value": str(alert['timestamp']), "short": False},
                        {"title": "Message", "value": alert['message'], "short": False}
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")

class ProductionMonitoringSystem:
    """Complete production monitoring system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Start Prometheus metrics server
        prometheus_client.start_http_server(8000)
        
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Set up default alert rules"""
        # Critical alerts
        self.alert_manager.add_alert_rule(
            "high_drawdown",
            lambda m: m.get('drawdown_percent', 0) > 15.0,
            "critical",
            ["email", "slack"],
            suppression_minutes=30
        )
        
        self.alert_manager.add_alert_rule(
            "component_failure",
            lambda m: any(status == 0 for status in m.get('component_statuses', {}).values()),
            "critical",
            ["email", "slack"],
            suppression_minutes=5
        )
        
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            lambda m: m.get('error_rate_per_minute', 0) > 10,
            "critical",
            ["email", "slack"],
            suppression_minutes=10
        )
        
        # Warning alerts
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            lambda m: m.get('memory_usage_percent', 0) > 85.0,
            "warning",
            ["slack"],
            suppression_minutes=15
        )
        
        self.alert_manager.add_alert_rule(
            "data_latency",
            lambda m: m.get('data_latency_seconds', 0) > 60,
            "warning",
            ["slack"],
            suppression_minutes=10
        )
        
        self.alert_manager.add_alert_rule(
            "low_cache_hit_rate",
            lambda m: m.get('cache_hit_rate', 1.0) < 0.8,
            "warning",
            ["slack"],
            suppression_minutes=20
        )
    
    def start_monitoring(self):
        """Start monitoring system"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts(current_metrics)
                
                # Update system metrics
                self._update_system_metrics()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu_usage_percent': cpu_percent,
            'memory_usage_percent': memory.percent,
            'memory_usage_bytes': memory.used,
            'disk_usage_percent': (disk.used / disk.total) * 100,
            'timestamp': datetime.now()
        }
        
        # Update Prometheus metrics
        self.metrics_collector.cpu_usage.set(cpu_percent)
        self.metrics_collector.memory_usage.set(memory.used)
        self.metrics_collector.disk_usage.set(metrics['disk_usage_percent'])
        
        return metrics
    
    def _update_system_metrics(self):
        """Update system-level metrics"""
        # This would integrate with your system components
        # to get component health status, error rates, etc.
        pass

# Production monitoring configuration
class MonitoringCapability(Capability):
    """Adds production monitoring to components"""
    
    def get_name(self) -> str:
        return "production_monitoring"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        monitoring_config = spec.get('monitoring', {})
        
        # Add metrics collection
        if not hasattr(component, '_metrics_collector'):
            component._metrics_collector = MetricsCollector()
        
        # Wrap methods with monitoring
        self._wrap_methods_with_monitoring(component, monitoring_config)
        
        return component
    
    def _wrap_methods_with_monitoring(self, component: Any, config: Dict[str, Any]):
        """Wrap component methods with monitoring"""
        monitored_methods = config.get('monitored_methods', [])
        
        for method_name in monitored_methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                
                def create_monitored_method(orig_method, method_name):
                    def monitored_method(*args, **kwargs):
                        start_time = time.time()
                        try:
                            result = orig_method(*args, **kwargs)
                            # Record success metric
                            processing_time = time.time() - start_time
                            component._metrics_collector.record_event_processing(
                                f"{component.__class__.__name__}.{method_name}", 
                                processing_time
                            )
                            return result
                        except Exception as e:
                            # Record error metric
                            component._metrics_collector.record_error(
                                component.__class__.__name__,
                                e.__class__.__name__
                            )
                            raise
                    return monitored_method
                
                setattr(component, method_name, create_monitored_method(original_method, method_name))
```

### 2.2 Monitoring Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'admf-trader'
    static_configs:
      - targets: ['admf-trader:8000']
    metrics_path: /metrics
    scrape_interval: 5s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

```yaml
# monitoring/alert_rules.yml
groups:
- name: admf_alerts
  rules:
  - alert: HighDrawdown
    expr: admf_drawdown_percent > 15
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "High portfolio drawdown detected"
      description: "Portfolio drawdown is {{ $value }}%, exceeding 15% threshold"

  - alert: ComponentDown
    expr: admf_component_status == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Component {{ $labels.component }} is down"
      description: "Component {{ $labels.component }} health check failing"

  - alert: HighErrorRate
    expr: rate(admf_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate in {{ $labels.component }}"
      description: "Error rate is {{ $value }} errors/second in {{ $labels.component }}"

  - alert: DataLatency
    expr: admf_data_latency_seconds > 60
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "High data latency from {{ $labels.source }}"
      description: "Data latency is {{ $value }} seconds from {{ $labels.source }}"

  - alert: LowCacheHitRate
    expr: rate(admf_cache_hits_total[5m]) / (rate(admf_cache_hits_total[5m]) + rate(admf_cache_misses_total[5m])) < 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low cache hit rate for {{ $labels.cache_name }}"
      description: "Cache hit rate is {{ $value }} for {{ $labels.cache_name }}"
```

---

## 3. Error Recovery Strategies

### 3.1 Resilient System Design

```python
from typing import Callable, Any, Dict, List, Optional, Type
import time
import random
import threading
from enum import Enum
from dataclasses import dataclass
from abc import abstractmethod

class RecoveryStrategy(Enum):
    RESTART_COMPONENT = "restart_component"
    RETRY_OPERATION = "retry_operation"
    SWITCH_TO_BACKUP = "switch_to_backup"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class RecoveryAction:
    """Recovery action definition"""
    strategy: RecoveryStrategy
    max_attempts: int
    backoff_strategy: str  # 'exponential', 'linear', 'fixed'
    base_delay_seconds: float
    max_delay_seconds: float
    success_threshold: int  # Success count needed to consider recovery successful

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()
    
    def __call__(self, func):
        """Decorator for circuit breaker"""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class RetryManager:
    """Advanced retry mechanism with multiple strategies"""
    
    def __init__(self):
        self.retry_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_retry_config(self, operation_name: str, config: Dict[str, Any]):
        """Register retry configuration for an operation"""
        self.retry_configs[operation_name] = config
    
    def retry_with_backoff(self, func: Callable, operation_name: str = None, 
                          *args, **kwargs) -> Any:
        """Retry function with configured backoff strategy"""
        config = self.retry_configs.get(operation_name, {
            'max_attempts': 3,
            'backoff_strategy': 'exponential',
            'base_delay': 1.0,
            'max_delay': 60.0,
            'jitter': True
        })
        
        max_attempts = config['max_attempts']
        backoff_strategy = config['backoff_strategy']
        base_delay = config['base_delay']
        max_delay = config['max_delay']
        jitter = config.get('jitter', True)
        
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == max_attempts - 1:
                    # Last attempt failed
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt, backoff_strategy, base_delay, max_delay, jitter)
                time.sleep(delay)
        
        raise RetryExhaustedException(f"Max retry attempts ({max_attempts}) exceeded", last_exception)
    
    def _calculate_delay(self, attempt: int, strategy: str, base_delay: float, 
                        max_delay: float, jitter: bool) -> float:
        """Calculate delay for next retry attempt"""
        if strategy == 'exponential':
            delay = base_delay * (2 ** attempt)
        elif strategy == 'linear':
            delay = base_delay * (attempt + 1)
        elif strategy == 'fixed':
            delay = base_delay
        else:
            delay = base_delay
        
        # Apply maximum delay limit
        delay = min(delay, max_delay)
        
        # Add jitter to avoid thundering herd
        if jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay

class FailoverManager:
    """Manages failover between primary and backup systems"""
    
    def __init__(self):
        self.primary_services: Dict[str, Any] = {}
        self.backup_services: Dict[str, Any] = {}
        self.current_active: Dict[str, str] = {}  # service_name -> 'primary' or 'backup'
        self.failover_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
    
    def register_service(self, service_name: str, primary: Any, backup: Any):
        """Register primary and backup services"""
        with self._lock:
            self.primary_services[service_name] = primary
            self.backup_services[service_name] = backup
            self.current_active[service_name] = 'primary'
    
    def get_active_service(self, service_name: str) -> Any:
        """Get currently active service"""
        with self._lock:
            active_type = self.current_active.get(service_name, 'primary')
            
            if active_type == 'primary':
                return self.primary_services.get(service_name)
            else:
                return self.backup_services.get(service_name)
    
    def failover_to_backup(self, service_name: str, reason: str = "") -> bool:
        """Failover to backup service"""
        with self._lock:
            if service_name not in self.backup_services:
                return False
            
            if self.current_active.get(service_name) == 'backup':
                return True  # Already on backup
            
            try:
                # Stop primary service if possible
                primary = self.primary_services[service_name]
                if hasattr(primary, 'stop'):
                    primary.stop()
                
                # Start backup service
                backup = self.backup_services[service_name]
                if hasattr(backup, 'start'):
                    backup.start()
                
                # Switch active service
                self.current_active[service_name] = 'backup'
                
                # Record failover
                self.failover_history.append({
                    'service_name': service_name,
                    'timestamp': datetime.now(),
                    'direction': 'primary_to_backup',
                    'reason': reason,
                    'success': True
                })
                
                return True
                
            except Exception as e:
                # Record failed failover
                self.failover_history.append({
                    'service_name': service_name,
                    'timestamp': datetime.now(),
                    'direction': 'primary_to_backup',
                    'reason': reason,
                    'success': False,
                    'error': str(e)
                })
                return False
    
    def failback_to_primary(self, service_name: str) -> bool:
        """Failback to primary service"""
        with self._lock:
            if self.current_active.get(service_name) == 'primary':
                return True  # Already on primary
            
            try:
                # Health check primary service
                primary = self.primary_services[service_name]
                if hasattr(primary, 'health_check'):
                    if not primary.health_check().status == HealthStatus.HEALTHY:
                        return False
                
                # Start primary service
                if hasattr(primary, 'start'):
                    primary.start()
                
                # Stop backup service
                backup = self.backup_services[service_name]
                if hasattr(backup, 'stop'):
                    backup.stop()
                
                # Switch active service
                self.current_active[service_name] = 'primary'
                
                # Record failback
                self.failover_history.append({
                    'service_name': service_name,
                    'timestamp': datetime.now(),
                    'direction': 'backup_to_primary',
                    'success': True
                })
                
                return True
                
            except Exception as e:
                self.failover_history.append({
                    'service_name': service_name,
                    'timestamp': datetime.now(),
                    'direction': 'backup_to_primary',
                    'success': False,
                    'error': str(e)
                })
                return False

class ErrorRecoveryOrchestrator:
    """Orchestrates error recovery across the system"""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        self.failover_manager = FailoverManager()
        self.recovery_history: List[Dict[str, Any]] = []
        
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Set up default recovery strategies"""
        # Database connection errors
        self.recovery_strategies['database_error'] = RecoveryAction(
            strategy=RecoveryStrategy.RETRY_OPERATION,
            max_attempts=3,
            backoff_strategy='exponential',
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            success_threshold=1
        )
        
        # Network timeout errors
        self.recovery_strategies['network_timeout'] = RecoveryAction(
            strategy=RecoveryStrategy.SWITCH_TO_BACKUP,
            max_attempts=2,
            backoff_strategy='fixed',
            base_delay_seconds=5.0,
            max_delay_seconds=5.0,
            success_threshold=1
        )
        
        # Component failures
        self.recovery_strategies['component_failure'] = RecoveryAction(
            strategy=RecoveryStrategy.RESTART_COMPONENT,
            max_attempts=3,
            backoff_strategy='linear',
            base_delay_seconds=10.0,
            max_delay_seconds=60.0,
            success_threshold=2
        )
        
        # Data quality issues
        self.recovery_strategies['data_quality_error'] = RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            max_attempts=1,
            backoff_strategy='fixed',
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            success_threshold=1
        )
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle error with appropriate recovery strategy"""
        error_type = self._classify_error(error, context)
        recovery_action = self.recovery_strategies.get(error_type)
        
        if not recovery_action:
            # No specific recovery strategy, use default retry
            return self._default_error_handling(error, context)
        
        try:
            success = self._execute_recovery(error_type, recovery_action, error, context)
            
            # Record recovery attempt
            self.recovery_history.append({
                'timestamp': datetime.now(),
                'error_type': error_type,
                'error_message': str(error),
                'recovery_strategy': recovery_action.strategy.value,
                'success': success,
                'context': context
            })
            
            return success
            
        except Exception as recovery_error:
            print(f"Recovery failed for {error_type}: {recovery_error}")
            return False
    
    def _classify_error(self, error: Exception, context: Dict[str, Any]) -> str:
        """Classify error type for recovery strategy selection"""
        error_class = error.__class__.__name__
        error_message = str(error).lower()
        
        # Database errors
        if 'database' in error_message or 'connection' in error_message:
            return 'database_error'
        
        # Network errors
        if 'timeout' in error_message or 'network' in error_message:
            return 'network_timeout'
        
        # Component errors
        if context.get('component_name'):
            return 'component_failure'
        
        # Data quality errors
        if 'data' in error_message and ('quality' in error_message or 'invalid' in error_message):
            return 'data_quality_error'
        
        return 'unknown_error'
    
    def _execute_recovery(self, error_type: str, recovery_action: RecoveryAction, 
                         error: Exception, context: Dict[str, Any]) -> bool:
        """Execute recovery strategy"""
        if recovery_action.strategy == RecoveryStrategy.RETRY_OPERATION:
            return self._retry_operation(recovery_action, context)
        
        elif recovery_action.strategy == RecoveryStrategy.RESTART_COMPONENT:
            return self._restart_component(recovery_action, context)
        
        elif recovery_action.strategy == RecoveryStrategy.SWITCH_TO_BACKUP:
            return self._switch_to_backup(recovery_action, context)
        
        elif recovery_action.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(recovery_action, context)
        
        elif recovery_action.strategy == RecoveryStrategy.FAILOVER:
            return self._perform_failover(recovery_action, context)
        
        else:
            return False
    
    def _retry_operation(self, recovery_action: RecoveryAction, context: Dict[str, Any]) -> bool:
        """Retry the failed operation"""
        operation = context.get('operation')
        if not operation:
            return False
        
        try:
            self.retry_manager.retry_with_backoff(operation, context.get('operation_name'))
            return True
        except:
            return False
    
    def _restart_component(self, recovery_action: RecoveryAction, context: Dict[str, Any]) -> bool:
        """Restart failed component"""
        component = context.get('component')
        if not component:
            return False
        
        try:
            if hasattr(component, 'stop'):
                component.stop()
            
            time.sleep(recovery_action.base_delay_seconds)
            
            if hasattr(component, 'start'):
                component.start()
            
            # Verify component health
            if hasattr(component, 'health_check'):
                health_result = component.health_check()
                return health_result.status == HealthStatus.HEALTHY
            
            return True
            
        except:
            return False
    
    def _switch_to_backup(self, recovery_action: RecoveryAction, context: Dict[str, Any]) -> bool:
        """Switch to backup service"""
        service_name = context.get('service_name')
        if not service_name:
            return False
        
        return self.failover_manager.failover_to_backup(service_name, "Error recovery")
    
    def _graceful_degradation(self, recovery_action: RecoveryAction, context: Dict[str, Any]) -> bool:
        """Implement graceful degradation"""
        degradation_handler = context.get('degradation_handler')
        if degradation_handler and callable(degradation_handler):
            try:
                degradation_handler()
                return True
            except:
                return False
        
        # Default degradation: continue with reduced functionality
        return True

class ResilientCapability(Capability):
    """Adds error recovery capabilities to components"""
    
    def get_name(self) -> str:
        return "resilient"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        resilience_config = spec.get('resilience', {})
        
        # Add circuit breaker for critical methods
        critical_methods = resilience_config.get('critical_methods', [])
        for method_name in critical_methods:
            if hasattr(component, method_name):
                self._add_circuit_breaker(component, method_name, resilience_config)
        
        # Add retry capability
        retry_methods = resilience_config.get('retry_methods', [])
        for method_name in retry_methods:
            if hasattr(component, method_name):
                self._add_retry_logic(component, method_name, resilience_config)
        
        # Add error recovery orchestrator
        if not hasattr(component, '_error_recovery'):
            component._error_recovery = ErrorRecoveryOrchestrator()
        
        return component
    
    def _add_circuit_breaker(self, component: Any, method_name: str, config: Dict[str, Any]):
        """Add circuit breaker to method"""
        original_method = getattr(component, method_name)
        circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('failure_threshold', 5),
            recovery_timeout=config.get('recovery_timeout', 60)
        )
        
        setattr(component, method_name, circuit_breaker(original_method))
    
    def _add_retry_logic(self, component: Any, method_name: str, config: Dict[str, Any]):
        """Add retry logic to method"""
        original_method = getattr(component, method_name)
        retry_manager = RetryManager()
        
        def retry_wrapper(*args, **kwargs):
            return retry_manager.retry_with_backoff(
                original_method, 
                f"{component.__class__.__name__}.{method_name}",
                *args, **kwargs
            )
        
        setattr(component, method_name, retry_wrapper)

# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass

class RetryExhaustedException(Exception):
    """Raised when max retry attempts are exhausted"""
    def __init__(self, message: str, last_exception: Exception):
        super().__init__(message)
        self.last_exception = last_exception
```

This covers the first parts of the production documentation. Would you like me to continue with Performance Tuning and Security Considerations to complete the production documentation?
