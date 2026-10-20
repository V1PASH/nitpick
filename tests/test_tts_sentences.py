"""Tests for the TTS sentence splitter."""
from __future__ import annotations

from nitpick.services.tts.openai import _split_sentences


def test_splits_on_sentence_end():
    sentences, remainder = _split_sentences("Hello there. How are you?")
    assert sentences == ["Hello there.", "How are you?"]
    assert remainder == ""


def test_keeps_an_incomplete_tail_buffered():
    sentences, remainder = _split_sentences("Hello there. How are")
    assert sentences == ["Hello there."]
    assert remainder == "How are"


def test_does_not_split_on_abbreviations():
    sentences, remainder = _split_sentences("Mr. Smith called. He left.")
    assert sentences == ["Mr. Smith called.", "He left."]
    assert remainder == ""


def test_handles_a_single_incomplete_sentence():
    sentences, remainder = _split_sentences("Hello")
    assert sentences == []
    assert remainder == "Hello"


def test_handles_empty_input():
    assert _split_sentences("") == ([], "")


def test_splits_exclamation_and_question_marks():
    sentences, _ = _split_sentences("Wow! Really? Yes.")
    assert sentences == ["Wow!", "Really?", "Yes."]
