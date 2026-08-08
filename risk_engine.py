"""
Risk Scoring Engine for Eagle Surveillance
Author: Bhagyashri
Issue: #19 - Multi-factor risk scoring algorithm
"""
import unittest

class RiskAnalyzer:
    def __init__(self):
        # Updated weights based on Issue #19 requirements
        self.weights = {
            "danger_zone": 0.35,       
            "restricted_zone": 0.35,   
            "suspicious_item": 0.20,
            "dwell_time": 0.20,
            "interaction": 0.25,
            "erratic_motion": 0.10,
            "repeated_approach": 0.10
        }

    def calculate_risk_score(
        self, 
        label: str, 
        zones_present: list[str],
        dwell_time_s: float = 0.0,
        min_dwell_time_s: float = 60.0,
        interaction_count: int = 0,
        max_interactions: int = 3,
        motion_type: str = "normal",
        repeated_approach_count: int = 0
    ) -> float:
        """
        Calculates a normalized risk score from 0.0 to 1.0 based on behavioral signals.
        """
        total_risk = 0.0
        unauthorized_zone = ("danger" in zones_present) or ("restricted" in zones_present)

        # 1. Zone Check
        if unauthorized_zone:
            total_risk += self.weights["restricted_zone"]

        # 2. Suspicious Items (Only risky if in unauthorized zones)
        if unauthorized_zone and label in ["backpack", "handbag", "suitcase"]:
            total_risk += self.weights["suspicious_item"]

        # 3. Dwell Time (How long they stayed)
        if dwell_time_s > min_dwell_time_s:
            total_risk += self.weights["dwell_time"]

        # 4. Interaction Count
        if interaction_count > max_interactions:
            total_risk += self.weights["interaction"]

        # 5. Motion Type
        if motion_type.lower() == "erratic":
            total_risk += self.weights["erratic_motion"]

        # 6. Repeated Approaches
        if repeated_approach_count > 2:
            total_risk += self.weights["repeated_approach"]

        # Normalize to [0.0, 1.0] and round to 2 decimal places
        final_score = round(min(max(total_risk, 0.0), 1.0), 2)
        return final_score

# --- Professional Unit Tests ---
class TestRiskAnalyzer(unittest.TestCase):
    """Test suite to verify risk score combinations."""
    
    def setUp(self):
        self.analyzer = RiskAnalyzer()

    def test_safe_zone_person(self):
        score = self.analyzer.calculate_risk_score("person", ["safe"])
        self.assertEqual(score, 0.0)

    def test_restricted_zone_backpack(self):
        score = self.analyzer.calculate_risk_score("backpack", ["restricted"])
        self.assertEqual(score, 0.55) # 0.35 (zone) + 0.20 (item)

    def test_erratic_motion_and_dwell(self):
        score = self.analyzer.calculate_risk_score(
            "person", ["danger"], 
            dwell_time_s=100.0, 
            motion_type="erratic"
        )
        self.assertEqual(score, 0.65) # 0.35 + 0.20 + 0.10

    def test_max_clamping(self):
        score = self.analyzer.calculate_risk_score(
            "backpack", ["danger"], 
            dwell_time_s=100.0, 
            interaction_count=5,
            motion_type="erratic",
            repeated_approach_count=3
        )
        self.assertEqual(score, 1.0) # Exceeds 1.0, should clamp to 1.0

if __name__ == '__main__':
    unittest.main()