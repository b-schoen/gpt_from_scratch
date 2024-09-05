# note: bubblewrap likely safer choice

# TODO(bschoen): See if we can access these from python
#
# Very useful: https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html
#

set -e # Exit immediately if a command exits with a non-zero status
set -x # Print commands and their arguments as they are executed

# TODO(bschoen): seccomp=my-seccomp.json ?
# TODO(bschoen): Run as non-root user: Create a non-privileged user and run your container processes as that user.
# TODO(bschoen): There's no disk I/O limit set, which could be a vector for resource exhaustion attacks.
# TODO(bschoen): There's no explicit network rate limiting, which could allow for network-based DoS attacks.
# TODO(bschoen): There's no health check defined, which could make it difficult to monitor the container's status.

#       --health-cmd string                Command to run to check health
#       --health-interval duration         Time between running the check (ms|s|m|h) (default 0s)
#       --health-retries int               Consecutive failures needed to report unhealthy
#       --health-start-interval duration   Time between running the check during the start period (ms|s|m|h) (default 0s)
#       --health-start-period duration     Start period for the container to initialize before starting health-retries
#                                          countdown (ms|s|m|h) (default 0s)
#       --health-timeout duration          Maximum time to allow one check to run (ms|s|m|h) (default 0s)

# TODO(bschoen): IO limitation
#
#   --device list                      Add a host device to the container
#   --device-cgroup-rule list          Add a rule to the cgroup allowed devices list
#   --device-read-bps list             Limit read rate (bytes per second) from a device (default [])
#   --device-read-iops list            Limit read rate (IO per second) from a device (default [])
#   --device-write-bps list            Limit write rate (bytes per second) to a device (default [])
#   --device-write-iops list           Limit write rate (IO per second) to a device (default [])

# TODO(bschoen): CPU limitations
#
#       --cpu-count int                    CPU count (Windows only)
#       --cpu-percent int                  CPU percent (Windows only)
#       --cpu-period int                   Limit CPU CFS (Completely Fair Scheduler) period
#       --cpu-quota int                    Limit CPU CFS (Completely Fair Scheduler) quota
#       --cpu-rt-period int                Limit CPU real-time period in microseconds
#       --cpu-rt-runtime int               Limit CPU real-time runtime in microseconds
#   -c, --cpu-shares int                   CPU shares (relative weight)
#       --cpus decimal                     Number of CPUs
#       --cpuset-cpus string               CPUs in which to allow execution (0-3, 0,1)
#       --cpuset-mems string               MEMs in which to allow execution (0-3, 0,1)

# TODO(bschoen): Can num proc be lower? `--pids-limit=1`?

# Run the evaluation container
#     --it \
#     --name my_untrusted_client_1 \         # Assign a name to the container for easy reference
# --network isolated_untrusted_client_network \            # Connect the container to the isolated network
# --publish 127.0.0.1:5555:5555 \            # Map container's port 5555 to host's localhost:5555
# --cap-drop=ALL \                           # Drop all Linux capabilities, reducing the container's privileges
# --security-opt="no-new-privileges:true" \  # Prevent processes from gaining additional privileges
# --cpus=1 \                                 # Limit the container to use 1 CPU core
# --memory=1g \                              # Limit the container's memory usage to 1 gigabyte
# --read-only \                              # Mount the container's root filesystem as read-only
# --tmpfs /tmp \                             # Create a temporary filesystem mounted at /tmp
# --ulimit nproc=1024 \                      # Limit number of processes
# untrusted_client                           # Use the 'untrusted_client' image to create this container
#
# REMOVED (WHY)
#  - --cap-drop=ALL \
docker run \
    --tty \
    --rm \
    --interactive \
    --name my_untrusted_client_1 \
    --network isolated_untrusted_client_network \
    --publish 127.0.0.1:5555:5555 \
    --security-opt="no-new-privileges:true" \
    --cpus=1 \
    --memory=1g \
    --read-only \
    --tmpfs /tmp \
    --ulimit nproc=1024 \
    untrusted_client
