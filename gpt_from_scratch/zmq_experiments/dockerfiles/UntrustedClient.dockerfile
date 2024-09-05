# Use a minimal base image with security updates
FROM python:3.10-slim-bullseye

# note: see `docker scout quickview` for additional vulnerabilities
# TODO(bschoen): Would a model just see these and blow through the open critical exploits?

# Set working directory
WORKDIR /app

# Install system dependencies and security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    netbase \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pyzmq pydantic

# Create a non-root user with minimal privileges and secure the environment
# Create user without home directory and with nologin shell
# RUN useradd -r -s /sbin/nologin -M appuser && \  
#     # Set root as owner of /app
#     chown -R root:root /app && \  
#     # Set read and execute permissions for owner and group, no permissions for others
#     chmod -R 550 /app && \  
#     # Create a temporary directory for runtime data
#     mkdir -p /app/tmp && \  
#     # Set appuser as owner of tmp directory
#     chown appuser:appuser /app/tmp && \
#     # Only the appuser can read, write, and execute in the tmp directory
#     chmod 700 /app/tmp && \
#     # Set more restrictive umask
#     echo "umask 027" >> /etc/bash.bashrc && \
#     # Disable core dumps
#     echo "* hard core 0" >> /etc/security/limits.conf && \
#     # Set more secure file permissions
#     find /app -type f -exec chmod 440 {} \; && \
#     find /app -type d -exec chmod 550 {} \;

# # move to tmp directory
# WORKDIR /app/tmp

# TODO(bschoen): This is pretty flexible
# Copy necessary Python files to the Docker container
COPY zmq_experiments zmq_experiments

# TODO(bschoen): Note if model can edit python this gives arbitrary code execution
# 
# Set appropriate permissions for copied files
# RUN chmod 550 client.py constants.py types.py && \
#     chown appuser:appuser client.py constants.py types.py

# # Switch to non-root user
# USER appuser

# Set secure environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

# Run the evaluation script with minimal permissions
# CMD ["ls"]
CMD ["python", "-m", "zmq_experiments.client"]
