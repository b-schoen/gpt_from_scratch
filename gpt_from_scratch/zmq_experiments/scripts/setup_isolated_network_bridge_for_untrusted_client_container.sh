# note: only needs to be run once

set -e # Exit immediately if a command exits with a non-zero status
set -x # Print commands and their arguments as they are executed

# Create a new Docker network named 'isolated_ai_network'

# - Use the 'bridge' network driver, which is the default for user-defined networks
# - Make this an internal network, preventing containers on this network from accessing the external network
# - The name of the network we're creating
docker network create \
    --driver bridge \
    --internal \
    isolated_untrusted_client_network
