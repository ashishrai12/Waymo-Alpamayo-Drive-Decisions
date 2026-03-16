"""
Unit tests for the DecisionSchema validation logic.
"""

import json
import pytest
from alpamayo_demo.core.schema import validate_decision, DECISION_SCHEMA


# --- Helpers ---

def make_valid_decision(**overrides):
    """Return a minimal valid decision dict, with optional overrides."""
    base = {
        "frame_id": 0,
        "scene_type": "intersection",
        "agents": [],
        "traffic_light": "green",
        "hazards": [],
        "decision": "maintain_speed",
        "confidence": 0.85,
        "reason": "Clear road ahead.",
    }
    base.update(overrides)
    return base


def to_json(d):
    return json.dumps(d)


# --- Valid Input Tests ---

class TestValidDecision:
    def test_valid_minimal_decision(self):
        d = make_valid_decision()
        result = validate_decision(to_json(d))
        assert result["decision"] == "maintain_speed"

    def test_all_scene_types_accepted(self):
        for scene in DECISION_SCHEMA["properties"]["scene_type"]["enum"]:
            d = make_valid_decision(scene_type=scene)
            result = validate_decision(to_json(d))
            assert result["scene_type"] == scene

    def test_all_traffic_lights_accepted(self):
        for light in DECISION_SCHEMA["properties"]["traffic_light"]["enum"]:
            d = make_valid_decision(traffic_light=light)
            result = validate_decision(to_json(d))
            assert result["traffic_light"] == light

    def test_all_decisions_accepted(self):
        for action in DECISION_SCHEMA["properties"]["decision"]["enum"]:
            d = make_valid_decision(decision=action)
            result = validate_decision(to_json(d))
            assert result["decision"] == action

    def test_confidence_boundary_values(self):
        for conf in [0.0, 0.5, 1.0]:
            d = make_valid_decision(confidence=conf)
            result = validate_decision(to_json(d))
            assert result["confidence"] == conf

    def test_multiple_agents_with_valid_types(self):
        agents = [
            {"type": "vehicle", "position": "ahead"},
            {"type": "pedestrian", "position": "crossing"},
            {"type": "cyclist", "position": "right"},
        ]
        d = make_valid_decision(agents=agents)
        result = validate_decision(to_json(d))
        assert len(result["agents"]) == 3

    def test_multiple_hazards(self):
        d = make_valid_decision(hazards=["construction", "weather", "oncoming vehicle"])
        result = validate_decision(to_json(d))
        assert len(result["hazards"]) == 3

    def test_returns_dict(self):
        result = validate_decision(to_json(make_valid_decision()))
        assert isinstance(result, dict)


# --- Invalid Input Tests ---

class TestInvalidDecision:
    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            validate_decision("not valid json {{{")

    def test_missing_required_field_raises(self):
        d = make_valid_decision()
        del d["decision"]
        with pytest.raises(ValueError, match="Missing required field"):
            validate_decision(to_json(d))

    def test_invalid_scene_type_raises(self):
        d = make_valid_decision(scene_type="highway")
        with pytest.raises(ValueError, match="Invalid scene_type"):
            validate_decision(to_json(d))

    def test_invalid_traffic_light_raises(self):
        d = make_valid_decision(traffic_light="purple")
        with pytest.raises(ValueError, match="Invalid traffic_light"):
            validate_decision(to_json(d))

    def test_invalid_decision_raises(self):
        d = make_valid_decision(decision="fly")
        with pytest.raises(ValueError, match="Invalid decision"):
            validate_decision(to_json(d))

    def test_confidence_above_1_raises(self):
        d = make_valid_decision(confidence=1.01)
        with pytest.raises(ValueError, match="Confidence"):
            validate_decision(to_json(d))

    def test_confidence_below_0_raises(self):
        d = make_valid_decision(confidence=-0.01)
        with pytest.raises(ValueError, match="Confidence"):
            validate_decision(to_json(d))

    def test_invalid_agent_type_raises(self):
        d = make_valid_decision(agents=[{"type": "drone", "position": "ahead"}])
        with pytest.raises(ValueError, match="Invalid agent type"):
            validate_decision(to_json(d))

    def test_invalid_agent_position_raises(self):
        d = make_valid_decision(agents=[{"type": "vehicle", "position": "behind"}])
        with pytest.raises(ValueError, match="Invalid agent position"):
            validate_decision(to_json(d))

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_decision("")

    def test_empty_json_object_raises(self):
        with pytest.raises(ValueError, match="Missing required field"):
            validate_decision("{}")
