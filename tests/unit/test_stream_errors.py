"""What a streaming failure says to the person reading it.

A model that writes a tool call the app cannot parse is the model's mistake,
not the reader's. The raw error is a JSON blob with a column number in it;
the message that reaches the transcript has to say what happened and what to
do about it.
"""

from tokirecur.api.app import _tool_call_parse_failure


def test_a_malformed_tool_call_is_recognised():
    error = Exception(
        'llama-server 500: {"error":{"code":500,"message":"Failed to parse tool '
        'call arguments as JSON: [json.exception.parse_error.101] parse error at '
        "line 1, column 5780: syntax error while parsing value - invalid string: "
        "missing closing quote\"}}"
    )
    assert _tool_call_parse_failure(error) is True


def test_an_ordinary_error_is_not_a_malformed_tool_call():
    assert _tool_call_parse_failure(Exception("connection reset by peer")) is False
    assert _tool_call_parse_failure(Exception("context overflow")) is False
