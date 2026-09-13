import numpy as np
import pytest
from src.pipeline import SafetyRulesEngine

def test_danger_zone_containment():
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    engine = SafetyRulesEngine(poly, dwell_threshold_seconds=1.0)
    assert engine.is_inside((50, 50)) == True
    assert engine.is_inside((200, 200)) == False

def test_dwell_time_trigger():
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    engine = SafetyRulesEngine(poly, dwell_threshold_seconds=0.1)
    
    # First entry
    inside, alert = engine.evaluate(track_id=1, foot_point=(50, 50))
    assert inside == True
    assert alert == False
