"""
Unit tests for visualization utility functions.

Tests the pure utility functions (wrap_text, create_display_frame)
without requiring a display / GUI.
"""

import numpy as np
import pytest
from alpamayo_demo.utils.visualization import wrap_text, create_display_frame


# --- wrap_text Tests ---

class TestWrapText:
    def test_short_text_not_wrapped(self):
        lines = wrap_text("Hello world", 50)
        assert lines == ["Hello world"]

    def test_long_text_is_wrapped(self):
        text = "This is a very long sentence that definitely exceeds the limit"
        lines = wrap_text(text, 20)
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 20 + len(line.split()[-1])  # allow for last word overflow

    def test_empty_string_returns_empty_list(self):
        lines = wrap_text("", 20)
        assert lines == []

    def test_single_word_returns_single_line(self):
        lines = wrap_text("Autonomous", 5)
        assert lines == ["Autonomous"]

    def test_exact_max_chars_not_wrapped(self):
        text = "Hello"
        lines = wrap_text(text, 5)
        assert len(lines) == 1

    def test_multiple_wraps(self):
        text = "a b c d e f g h"
        lines = wrap_text(text, 4)
        assert len(lines) >= 4

    def test_preserves_all_words(self):
        text = "one two three four five"
        lines = wrap_text(text, 10)
        combined = " ".join(lines)
        for word in text.split():
            assert word in combined


# --- create_display_frame Tests ---

def make_minimal_decision(**overrides):
    base = {
        "frame_id": 0,
        "scene_type": "intersection",
        "agents": [],
        "traffic_light": "red",
        "hazards": [],
        "decision": "stop",
        "confidence": 0.92,
        "reason": "Red light detected.",
    }
    base.update(overrides)
    return base


class TestCreateDisplayFrame:
    def test_output_is_numpy_array(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = create_display_frame(frame, make_minimal_decision())
        assert isinstance(result, np.ndarray)

    def test_output_has_3_channels(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = create_display_frame(frame, make_minimal_decision())
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_output_wider_than_input(self):
        """Display frame should include info panel — so wider than original."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = create_display_frame(frame, make_minimal_decision())
        assert result.shape[1] > 640

    def test_with_agents(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d = make_minimal_decision(agents=[
            {"type": "pedestrian", "position": "crossing"},
            {"type": "vehicle", "position": "ahead"},
        ])
        result = create_display_frame(frame, d)
        assert result is not None

    def test_with_hazards(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d = make_minimal_decision(hazards=["construction", "weather"])
        result = create_display_frame(frame, d)
        assert isinstance(result, np.ndarray)

    def test_with_small_frame(self):
        """Small frames shouldn't crash — they get scaled down."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = create_display_frame(frame, make_minimal_decision())
        assert result is not None

    def test_with_large_frame(self):
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
        result = create_display_frame(frame, make_minimal_decision())
        assert result is not None

    def test_long_reason_text_handled(self):
        """Very long reason text should be wrapped without crashing."""
        long_reason = "The vehicle detected an unusual scenario " * 5
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        d = make_minimal_decision(reason=long_reason)
        result = create_display_frame(frame, d)
        assert isinstance(result, np.ndarray)
