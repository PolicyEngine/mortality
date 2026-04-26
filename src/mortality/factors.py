"""Personal factors affecting mortality, loaded from research data."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml


@lru_cache(maxsize=1)
def load_research_factors() -> Dict[str, Any]:
    """Load research-based mortality factors from YAML file."""
    factors_file = Path(__file__).parent / "research" / "factors.yaml"

    with open(factors_file, "r") as f:
        return yaml.safe_load(f)


@dataclass
class MortalityFactors:
    """Personal factors that affect mortality.

    All factors and their effects are based on peer-reviewed research,
    loaded from research/factors.yaml.
    """

    # Demographics
    gender: Literal["male", "female"] = "male"

    # Health status
    health: Literal["excellent", "good", "average", "poor"] = "average"
    smoker: bool = False
    bmi: Optional[float] = None

    # Socioeconomic factors
    education: Optional[
        Literal[
            "less_than_high_school",
            "high_school",
            "some_college",
            "bachelors",
            "graduate",
        ]
    ] = None
    income_percentile: Optional[int] = None  # 1-100

    # Lifestyle
    exercise: Optional[Literal["none", "light", "moderate", "vigorous"]] = None
    alcohol: Optional[Literal["none", "light", "moderate", "heavy"]] = None

    def get_multiplier(self) -> float:
        """Calculate mortality multiplier based on research data.

        Returns:
            Combined multiplier for all factors (1.0 = average mortality)
        """
        research = load_research_factors()
        multiplier = 1.0

        # Health status
        health_data = research["health_status"]
        multiplier *= health_data["multipliers"][self.health]

        # Smoking
        if self.smoker:
            smoking_data = research["smoking"]
            multiplier *= smoking_data["multipliers"]["current_smoker"]

        # BMI
        if self.bmi is not None:
            bmi_data = research["bmi"]
            for range_data in bmi_data["ranges"]:
                min_bmi, max_bmi = range_data["range"]
                if min_bmi <= self.bmi < max_bmi:
                    multiplier *= range_data["multiplier"]
                    break

        # Income
        if self.income_percentile is not None:
            income_data = research["income"]
            for range_data in income_data["percentile_ranges"]:
                min_pct, max_pct = range_data["range"]
                if min_pct <= self.income_percentile <= max_pct:
                    multiplier *= range_data["multiplier"]
                    break

        # Education
        if self.education is not None:
            education_data = research["education"]
            multiplier *= education_data["multipliers"][self.education]

        # Exercise
        if self.exercise is not None:
            exercise_data = research["exercise"]
            multiplier *= exercise_data["multipliers"][self.exercise]

        # Alcohol
        if self.alcohol is not None:
            alcohol_data = research["alcohol"]
            multiplier *= alcohol_data["multipliers"][self.alcohol]

        # Apply interaction effects
        interactions = research.get("interactions", {})

        # Smoking + exercise interaction
        if self.smoker and self.exercise in ["moderate", "vigorous"]:
            if "smoking_and_exercise" in interactions:
                multiplier *= interactions["smoking_and_exercise"]["effect"]

        # Income + education interaction (avoid double counting)
        if self.income_percentile is not None and self.education is not None:
            if "income_and_education" in interactions:
                multiplier *= interactions["income_and_education"]["effect"]

        return multiplier

    def get_citations(self) -> Dict[str, Dict[str, Any]]:
        """Get citations for all factors being used.

        Returns:
            Dictionary of factor names to citation information
        """
        research = load_research_factors()
        citations = {}

        # Add citations for factors being used
        if self.health != "average":
            citations["health_status"] = research["health_status"]["source"]

        if self.smoker:
            citations["smoking"] = research["smoking"]["source"]

        if self.bmi is not None:
            citations["bmi"] = research["bmi"]["source"]

        if self.income_percentile is not None:
            citations["income"] = research["income"]["source"]

        if self.education is not None:
            citations["education"] = research["education"]["source"]

        if self.exercise is not None:
            citations["exercise"] = research["exercise"]["source"]

        if self.alcohol is not None:
            citations["alcohol"] = research["alcohol"]["source"]

        return citations
