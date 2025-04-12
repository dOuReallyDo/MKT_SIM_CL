"""
Configurazione per il monitoraggio delle performance del sistema
"""

# Configurazione Redis
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': None,
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'retry_on_timeout': True
}

# Configurazione Prometheus
PROMETHEUS_CONFIG = {
    'port': 8000,
    'host': '0.0.0.0',
    'metrics_path': '/metrics'
}

# Configurazione metriche
METRICS_CONFIG = {
    'trading_operations': {
        'name': 'trading_operations_total',
        'description': 'Numero totale di operazioni di trading',
        'labels': ['agent_id', 'symbol', 'action']
    },
    'cache_hits': {
        'name': 'cache_hits_total',
        'description': 'Numero di hit nella cache',
        'labels': ['cache_type']
    },
    'cache_misses': {
        'name': 'cache_misses_total',
        'description': 'Numero di miss nella cache',
        'labels': ['cache_type']
    },
    'memory_usage': {
        'name': 'memory_usage_bytes',
        'description': 'Uso della memoria in bytes',
        'labels': ['process']
    },
    'cpu_usage': {
        'name': 'cpu_usage_percent',
        'description': 'Uso della CPU in percentuale',
        'labels': ['process']
    },
    'operation_latency': {
        'name': 'operation_latency_seconds',
        'description': 'Latenza delle operazioni',
        'labels': ['operation_type'],
        'buckets': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    }
}

# Configurazione logging performance
PERFORMANCE_LOGGING = {
    'enabled': True,
    'log_file': 'logs/performance.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'level': 'INFO'
}

# Configurazione alerting
ALERTING_CONFIG = {
    'enabled': True,
    'memory_threshold': 0.8,  # 80% della memoria disponibile
    'cpu_threshold': 0.9,     # 90% della CPU
    'latency_threshold': 5.0, # 5 secondi
    'notification_email': 'admin@example.com'
}

# Configurazione sampling
SAMPLING_CONFIG = {
    'memory_sampling_interval': 60,  # secondi
    'cpu_sampling_interval': 60,     # secondi
    'operation_sampling_rate': 1.0   # 100% delle operazioni
} 