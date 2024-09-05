# ex:
#
#       zmq_experiments git:(main) ✗ docker build --tag untrusted_client --file dockerfiles/UntrustedClient.dockerfile .

set -x
set -e

docker build --tag untrusted_client --file dockerfiles/UntrustedClient.dockerfile .
