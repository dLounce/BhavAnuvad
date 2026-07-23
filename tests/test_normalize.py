import pytest

from bhav.text.normalize import detect_script, normalize, normalize_for_scoring


def test_devanagari_digits_to_ascii():
    assert normalize("मैं २५ साल का हूं", "hi") == "मैं 25 साल का हूं"


def test_tamil_digits_to_ascii():
    assert "25" in normalize("௨௫ ஆண்டுகள்", "ta")


def test_danda_becomes_period():
    assert normalize("वह गया।", "hi").endswith(".")


def test_whitespace_collapsed():
    assert normalize("  मैं   ठीक  हूं  ", "hi") == "मैं ठीक हूं"


def test_smart_quotes_normalized():
    assert "'" in normalize("वह \u2018ठीक\u2019 है", "hi")


def test_zero_width_chars_stripped():
    assert "\u200b" not in normalize("ठी\u200bक", "hi")


def test_foreign_script_dropped():
    out = normalize("மாலை வணக்கம் नमस्ते", "ta")
    assert "नमस्ते" not in out
    assert "மாலை" in out


def test_scoring_strips_punctuation_and_case():
    assert normalize_for_scoring("Hello, वह गया।", "hi") == "hello वह गया"


def test_repeated_punctuation_collapsed():
    assert normalize("क्या?? सच!!", "hi") == "क्या? सच!"


def test_empty_string():
    assert normalize("", "hi") == ""


def test_bad_lang_raises():
    with pytest.raises(ValueError):
        normalize("test", "en")


@pytest.mark.parametrize(
    "text,expected",
    [("नमस्ते दोस्त", "hi"), ("வணக்கம் நண்பா", "ta"), ("hello world", None)],
)
def test_detect_script(text, expected):
    assert detect_script(text) == expected