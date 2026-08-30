"""Dependency-free tests for script parsing and chunking: python test_logic.py"""

from script_parser import Turn, normalize_speaker, parse_script, split_into_chunks


def test_basic_host_guest():
    script = (
        "HOST: Welcome to the show.\n"
        "GUEST: Thanks for having me.\n"
        "HOST: Let's begin."
    )
    turns = parse_script(script)
    assert [t.speaker for t in turns] == ["HOST", "GUEST", "HOST"], turns
    assert turns[0].text == "Welcome to the show."


def test_markdown_and_cues():
    script = (
        "**Host:** Big quarter, huh? [laughs]\n"
        "[transition music]\n"
        "*Guest*: Absolutely — revenue was up 18%.\n"
        "And margins improved too.\n"
    )
    turns = parse_script(script)
    assert [t.speaker for t in turns] == ["HOST", "GUEST"], turns
    assert "laughs" not in turns[0].text
    assert turns[1].text == "Absolutely — revenue was up 18%. And margins improved too."


def test_custom_speaker_names_and_aliases():
    script = (
        "Sarah: Hello everyone.\n"
        "Dr. Chen: Glad to be here.\n"
        "Speaker 1: Back to you.\n"
    )
    turns = parse_script(script)
    assert [t.speaker for t in turns] == ["SARAH", "DR. CHEN", "HOST"], turns
    assert normalize_speaker("Interviewer") == "HOST"
    assert normalize_speaker("analyst") == "GUEST"


def test_consecutive_same_speaker_merges():
    turns = parse_script("HOST: One.\nHOST: Two.")
    assert turns == [Turn("HOST", "One. Two.")]


def test_unlabeled_first_line_defaults_to_host():
    turns = parse_script("Just some narration text.")
    assert turns == [Turn("HOST", "Just some narration text.")]


def test_chunking_respects_sentences():
    text = "First sentence here. " * 30  # ~630 chars
    chunks = split_into_chunks(text.strip(), max_chars=100)
    assert all(len(c) <= 100 for c in chunks), [len(c) for c in chunks]
    assert " ".join(chunks).count("First sentence here.") == 30


def test_chunking_hard_splits_long_sentence():
    text = "word " * 120  # one 600-char "sentence", no periods
    chunks = split_into_chunks(text.strip(), max_chars=100)
    assert all(len(c) <= 100 for c in chunks), [len(c) for c in chunks]
    assert " ".join(chunks).split() == text.split()


def test_empty_script():
    assert parse_script("") == []
    assert parse_script("\n[music]\n") == []


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
