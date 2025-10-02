"""Sample objects for testing PathKeyResolverMixin functionality."""

# Simple dictionary for basic path resolution
SIMPLE_CONFIG = {"name": "test_model", "version": "1.0", "enabled": True}

# Nested dictionary for key-based access
NESTED_CONFIG = {
    "database": {"host": "localhost", "port": 5432, "name": "test_db"},
    "cache": {"ttl": 3600, "max_size": 1000},
    "features": {"feature_a": True, "feature_b": False},
}

# Complex nested structure
COMPLEX_CONFIG = {
    "environments": {
        "dev": {"database": {"host": "dev.example.com", "port": 5432}, "debug": True},
        "prod": {"database": {"host": "prod.example.com", "port": 5432}, "debug": False},
    },
    "shared": {"timeout": 30, "retries": 3},
}

# Additional test data with various types
MIXED_TYPES_CONFIG = {
    "string_val": "hello",
    "int_val": 42,
    "float_val": 3.14,
    "bool_val": True,
    "list_val": [1, 2, 3],
    "dict_val": {"nested": "value"},
}

# Alternate nested config to test Hydra overrides of `path`/`key`
OTHER_NESTED_CONFIG = {
    "database": {"host": "override.local", "port": 6543, "name": "other_db"},
    "database_alt": {"host": "alt.local", "port": 7777, "name": "alt_db"},
    "cache": {"ttl": 7200, "max_size": 2000},
    "features": {"feature_a": False, "feature_b": True},
}

# List of servers to test top-level integer key traversal
SERVERS = [
    {"host": "server1.local", "port": 1111, "name": "s1"},
    {"host": "server2.local", "port": 2222, "name": "s2"},
]
