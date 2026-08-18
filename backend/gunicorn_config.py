"""Gunicorn configuration for production deployment on Azure App Service."""

# Number of worker processes
workers = 4

# Worker class - using Uvicorn worker for async support
worker_class = "uvicorn.workers.UvicornWorker"

# Bind address and port
bind = "0.0.0.0:8000"

# Timeout for workers
timeout = 120

# Max requests before worker restart
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL/TLS - Handled by Azure App Service, not needed here
keyfile = None
certfile = None
