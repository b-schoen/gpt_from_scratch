import zmq

from . import message_types, constants

# TODO(bschoen): Would want heartbeat


def main() -> None:

    # Client-side example
    context = zmq.Context()
    client_socket = context.socket(zmq.REQ)
    client_socket.connect("tcp://localhost:5555")

    message = message_types.Message(content="Initial message")

    print(f"Sending initial message: {message}")
    client_socket.send(message.model_dump_json().encode())

    while True:

        print("Awaiting response...")
        response = client_socket.recv()

        print("Received response, parsing")
        parsed_response = message_types.Message.model_validate_json(response)

        print(f"Received response: {parsed_response}")


if __name__ == "__main__":
    main()
