from io import BytesIO

import pytest

from zoo_mcp.hosted.protocol import encode, read_message


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_worker_pipe_eof_mid_message(length):
    stream = BytesIO(encode({"tool": "example"})[:length])
    with pytest.raises(EOFError, match="Incomplete worker message"):
        read_message(stream)


def test_worker_pipe_messages_and_clean_eof():
    first = {"tool": "first"}
    second = {"tool": "second"}
    stream = BytesIO(encode(first) + encode(second))
    assert read_message(stream) == first
    assert read_message(stream) == second
    assert read_message(stream) is None
