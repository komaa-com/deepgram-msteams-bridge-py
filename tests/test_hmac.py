from deepgram_msteams_bridge.hmac_auth import is_fresh, sign, verify


def test_sign_is_deterministic_hex():
    sig = sign("secret", 1720000000000, "call-1")
    assert sig == sign("secret", 1720000000000, "call-1")
    assert len(sig) == 64
    assert sig == sig.lower()


def test_verify_roundtrip():
    ts = 1720000000000
    sig = sign("secret", ts, "call-1")
    assert verify("secret", ts, "call-1", sig)
    assert verify("secret", ts, "call-1", sig.upper())  # case-insensitive input


def test_verify_rejects_wrong_inputs():
    ts = 1720000000000
    sig = sign("secret", ts, "call-1")
    assert not verify("other", ts, "call-1", sig)
    assert not verify("secret", ts + 1, "call-1", sig)
    assert not verify("secret", ts, "call-2", sig)
    assert not verify("secret", ts, "call-1", "")
    assert not verify("", ts, "call-1", sig)
    assert not verify("secret", ts, "", sig)


def test_timestamp_string_matches_number():
    # the worker sends the timestamp as a header string; signing must agree
    assert sign("s", "123", "c") == sign("s", 123, "c")


def test_is_fresh_window():
    now = 1_000_000
    assert is_fresh(now, 60_000, now)
    assert is_fresh(now - 60_000, 60_000, now)
    assert is_fresh(now + 60_000, 60_000, now)
    assert not is_fresh(now - 60_001, 60_000, now)
    assert not is_fresh(float("nan"), 60_000, now)
    assert not is_fresh(float("inf"), 60_000, now)
