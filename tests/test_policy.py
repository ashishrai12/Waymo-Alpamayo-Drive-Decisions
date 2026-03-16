"""
Unit tests for AlpamayoPolicy — mock mode behavior and edge cases.
"""

import json
import numpy as np
import pytest
from alpamayo_demo.core.policy import AlpamayoPolicy


GOAL_PROMPT = "Analyze the scene and decide the next action."
VALID_DECISIONS = {"accelerate", "maintain_speed", "slow_down", "brake", "stop", "yield"}
VALID_SCENES = {"intersection", "straight_road", "crosswalk", "parking_lot"}
VALID_LIGHTS = {"red", "yellow", "green", "unknown"}


def blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestAlpamayoPolicyMock:
    def setup_method(self):
        self.policy = AlpamayoPolicy(mock=True)

    def test_decide_returns_string(self):
        result = self.policy.decide(blank_frame(), GOAL_PROMPT)
        assert isinstance(result, str)

    def test_decide_returns_valid_json(self):
        result = self.policy.decide(blank_frame(), GOAL_PROMPT)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_decision_key_is_valid_action(self):
        for _ in range(10):
            result = json.loads(self.policy.decide(blank_frame(), GOAL_PROMPT))
            assert result["decision"] in VALID_DECISIONS

    def test_scene_type_is_valid(self):
        for _ in range(10):
            result = json.loads(self.policy.decide(blank_frame(), GOAL_PROMPT))
            assert result["scene_type"] in VALID_SCENES

    def test_traffic_light_is_valid(self):
        for _ in range(10):
            result = json.loads(self.policy.decide(blank_frame(), GOAL_PROMPT))
            assert result["traffic_light"] in VALID_LIGHTS

    def test_confidence_in_range(self):
        for _ in range(10):
            result = json.loads(self.policy.decide(blank_frame(), GOAL_PROMPT))
            assert 0.0 <= result["confidence"] <= 1.0

    def test_reason_is_non_empty_string(self):
        result = json.loads(self.policy.decide(blank_frame(), GOAL_PROMPT))
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_agents_is_list(self):
        result = json.loads(self.policy.decide(blank_frame(), GOAL_PROMPT))
        assert isinstance(result["agents"], list)

    def test_hazards_is_list(self):
        result = json.loads(self.policy.decide(blank_frame(), GOAL_PROMPT))
        assert isinstance(result["hazards"], list)

    def test_works_with_various_frame_sizes(self):
        for size in [(240, 320), (480, 640), (720, 1280)]:
            frame = blank_frame(size[0], size[1])
            result = json.loads(self.policy.decide(frame, GOAL_PROMPT))
            assert "decision" in result

    def test_works_with_colored_frame(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = json.loads(self.policy.decide(frame, GOAL_PROMPT))
        assert "decision" in result

    def test_different_prompts_still_work(self):
        for prompt in ["Go fast", "", "Stop immediately", "Yield to all agents"]:
            result = json.loads(self.policy.decide(blank_frame(), prompt))
            assert "decision" in result

    def test_multiple_consecutive_decisions(self):
        """Policy should be stable across many calls."""
        results = [json.loads(self.policy.decide(blank_frame(), GOAL_PROMPT)) for _ in range(20)]
        assert all("decision" in r for r in results)
        assert all(r["decision"] in VALID_DECISIONS for r in results)


class TestAlpamayoPolicyReal:
    def test_real_mode_raises_not_implemented(self):
        policy = AlpamayoPolicy(mock=False)
        with pytest.raises(NotImplementedError):
            policy.decide(blank_frame(), GOAL_PROMPT)
