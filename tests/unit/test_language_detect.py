"""Unit tests for language/script detection (spec §11.1)."""
from src.intake.language_detect import detect_language, is_latin


def test_latin_english():
    assert detect_language("find the best laptop prices").family == "latin"


def test_latin_with_accented_chars():
    assert detect_language("meilleur ordinateur portable").family == "latin"
    assert detect_language("¿Cuál es el mejor?").family == "latin"


def test_arabic():
    assert detect_language("مرحبا بالعالم").family == "arabic"


def test_cjk_chinese():
    assert detect_language("你好世界").family == "cjk"


def test_cjk_japanese_hiragana():
    assert detect_language("こんにちは").family == "cjk"


def test_korean():
    assert detect_language("안녕하세요").family == "korean"


def test_cyrillic():
    assert detect_language("привет мир").family == "cyrillic"


def test_devanagari():
    assert detect_language("नमस्ते दुनिया").family == "devanagari"


def test_greek():
    assert detect_language("γειά σου").family == "greek"


def test_hebrew():
    assert detect_language("שלום עולם").family == "hebrew"


# --- edge cases ---------------------------------------------------------

def test_empty_string_is_unknown():
    assert detect_language("").family == "unknown"


def test_whitespace_only_is_unknown():
    assert detect_language("   \n\t  ").family == "unknown"


def test_digits_only_is_unknown():
    assert detect_language("12345").family == "unknown"


def test_punctuation_only_is_unknown():
    assert detect_language("!!! ???").family == "unknown"


# --- mixed --------------------------------------------------------------

def test_dominant_script_wins():
    g = detect_language("Hello 你好 你好 你好")
    assert g.family == "cjk"


def test_latin_dominant_wins():
    g = detect_language("hello world this is mostly english 你好")
    assert g.family == "latin"


def test_confidence_is_reported():
    g = detect_language("hello world")
    assert 0.0 < g.confidence <= 1.0


# --- is_latin helper ----------------------------------------------------

def test_is_latin_true_for_english():
    assert is_latin("best laptop prices")


def test_is_latin_false_for_arabic():
    assert not is_latin("مرحبا")