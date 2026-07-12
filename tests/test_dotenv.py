from deepgram_msteams_bridge import load_dotenv


def test_dotenv_parsing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "PLAIN=value",
                "export EXPORTED=from-shell-style",
                'QUOTED="keep # this"',
                "SINGLE='also kept'",
                "COMMENTED=value # trailing comment",
                "HASH_NO_SPACE=a#b",
                "EXISTING=overwritten?",
                "  SPACED  =  padded  ",
                "not a kv line",
            ]
        )
    )
    for key in ("PLAIN", "EXPORTED", "QUOTED", "SINGLE", "COMMENTED", "HASH_NO_SPACE", "SPACED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EXISTING", "kept")

    load_dotenv(str(env))

    import os

    assert os.environ["PLAIN"] == "value"
    assert os.environ["EXPORTED"] == "from-shell-style"  # `export ` prefix stripped
    assert os.environ["QUOTED"] == "keep # this"  # quoted values keep '#'
    assert os.environ["SINGLE"] == "also kept"
    assert os.environ["COMMENTED"] == "value"  # inline comment stripped
    assert os.environ["HASH_NO_SPACE"] == "a#b"  # '#' without space is part of the value
    assert os.environ["SPACED"] == "padded"
    assert os.environ["EXISTING"] == "kept"  # never overwrites


def test_dotenv_missing_file_is_fine(tmp_path):
    load_dotenv(str(tmp_path / "nope.env"))  # must not raise
