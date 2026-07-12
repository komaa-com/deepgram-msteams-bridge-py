from deepgram_msteams_bridge.protocol import parse_worker_message, pcm16k_bytes_to_ms


def test_parse_valid_message():
    msg = parse_worker_message('{"type":"ping","ts":123}')
    assert msg == {"type": "ping", "ts": 123}


def test_parse_bytes_input():
    msg = parse_worker_message(b'{"type":"session.end","reason":"x"}')
    assert msg is not None and msg["type"] == "session.end"


def test_parse_rejects_junk():
    assert parse_worker_message("not json") is None
    assert parse_worker_message("[1,2,3]") is None
    assert parse_worker_message('{"noType":true}') is None
    assert parse_worker_message('{"type":123}') is None
    assert parse_worker_message(b"\xff\xfe") is None


def test_pcm_duration():
    # 640 bytes = 20 ms at 16 kHz mono 16-bit
    assert pcm16k_bytes_to_ms(640) == 20
    assert pcm16k_bytes_to_ms(32000) == 1000
