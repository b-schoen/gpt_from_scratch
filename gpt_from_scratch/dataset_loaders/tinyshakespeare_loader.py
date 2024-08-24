from gpt_from_scratch import file_utils


def read_input_text() -> str:
    """Get the full `input_text` of the tinyshakespeare dataset."""

    # load tinyshakespeare
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    input_filepath = file_utils.download_file_from_url(url)

    # Read all text from the input file
    input_text = input_filepath.read_text()

    return input_text
