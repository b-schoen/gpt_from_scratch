from . import message_types, constants

import zmq


def main() -> None:

    # ZeroMQ setup
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")

    while True:
        # Receive message
        print("Waiting for message...")

        # TODO(bschoen): Want limitations on the size and speed of this connection
        zmq_message_bytes = socket.recv()

        # Deserialize and validate input
        message = message_types.Message.model_validate_json(zmq_message_bytes)

        print(f"Received message: {message}")

        message.content = f"Responded to: {message.content}"

        print(f"Responding with message: {message}")
        socket.send(message.model_dump_json().encode())

        print("Sent message")


if __name__ == "__main__":
    main()
