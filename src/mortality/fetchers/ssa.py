"""Fetch mortality data from Social Security Administration."""

import re
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup


class SSATableFetcher:
    """Fetch and cache SSA Period Life Tables."""

    BASE_URL = "https://www.ssa.gov/oact/STATS/table4c6.html"
    CACHE_DIR = Path.home() / ".cache" / "mortality" / "ssa"
    CACHE_DURATION = timedelta(days=30)  # Refresh monthly

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize SSA fetcher with optional cache directory."""
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self, force_refresh: bool = False
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Fetch SSA mortality tables, using cache if available.

        Args:
            force_refresh: Force download even if cache exists

        Returns:
            Tuple of (male_rates, female_rates) dictionaries
        """
        cache_file = self.cache_dir / "ssa_period_life_table.json"

        # Check cache
        if not force_refresh and self._is_cache_valid(cache_file):
            return self._load_cache(cache_file)

        # Fetch fresh data
        male_rates, female_rates = self._fetch_from_web()

        # Save to cache
        self._save_cache(cache_file, male_rates, female_rates)

        return male_rates, female_rates

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is recent."""
        if not cache_file.exists():
            return False

        # Check age
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - mtime

        return age < self.CACHE_DURATION

    def _load_cache(
        self, cache_file: Path
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Load mortality data from cache."""
        with open(cache_file, "r") as f:
            data = json.load(f)

        # Convert string keys back to integers
        male_rates = {int(k): v for k, v in data["male"].items()}
        female_rates = {int(k): v for k, v in data["female"].items()}

        return male_rates, female_rates

    def _save_cache(
        self,
        cache_file: Path,
        male_rates: Dict[int, float],
        female_rates: Dict[int, float],
    ) -> None:
        """Save mortality data to cache."""
        data = {
            "source": self.BASE_URL,
            "fetched": datetime.now().isoformat(),
            "male": {str(k): v for k, v in male_rates.items()},
            "female": {str(k): v for k, v in female_rates.items()},
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _fetch_from_web(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Fetch mortality tables from SSA website."""
        response = requests.get(self.BASE_URL)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find the table with mortality data
        # SSA page has a specific structure we need to parse
        male_rates = {}
        female_rates = {}

        # Look for the main data table
        # The SSA table has rows with: Age | Male Death Probability | Female Death Probability
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all("td")

                if len(cells) >= 7:  # SSA table has 7 columns
                    try:
                        # Extract age from first cell
                        age_text = cells[0].get_text().strip()
                        age_match = re.match(r"^(\d+)", age_text)

                        if age_match:
                            age = int(age_match.group(1))

                            # Male death probability is in column 2
                            male_text = cells[1].get_text().strip()
                            male_rate = self._parse_rate(male_text)

                            # Female death probability is in column 4
                            female_text = cells[4].get_text().strip()
                            female_rate = self._parse_rate(female_text)

                            if male_rate is not None and female_rate is not None:
                                male_rates[age] = male_rate
                                female_rates[age] = female_rate

                    except (ValueError, IndexError):
                        continue

        # If we didn't get data, use fallback
        if not male_rates:
            print("Warning: Could not parse SSA website, using fallback data")
            male_rates, female_rates = self._get_fallback_data()

        return male_rates, female_rates

    def _parse_rate(self, text: str) -> Optional[float]:
        """Parse a mortality rate from text."""
        # Remove commas and extra characters
        text = re.sub(r"[^\d.]", "", text)

        if text:
            try:
                return float(text)
            except ValueError:
                return None
        return None

    def _get_fallback_data(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Get fallback SSA 2021 data if web fetch fails."""
        # Minimal fallback data for key ages
        # In production, this would be more complete
        male = {
            0: 0.00598,
            10: 0.00014,
            20: 0.00104,
            30: 0.00187,
            40: 0.00322,
            50: 0.00533,
            60: 0.01158,
            65: 0.01604,
            70: 0.02476,
            75: 0.03843,
            80: 0.06069,
            85: 0.09764,
            90: 0.15829,
            95: 0.25457,
            100: 0.40032,
            110: 0.88489,
            120: 1.00000,
        }

        female = {
            0: 0.00493,
            10: 0.00011,
            20: 0.00036,
            30: 0.00063,
            40: 0.00128,
            50: 0.00324,
            60: 0.00748,
            65: 0.01052,
            70: 0.01642,
            75: 0.02653,
            80: 0.04365,
            85: 0.07247,
            90: 0.11991,
            95: 0.19620,
            100: 0.31467,
            110: 0.72299,
            120: 1.00000,
        }

        # Interpolate missing ages
        for rates in [male, female]:
            ages = sorted(rates.keys())
            for i in range(len(ages) - 1):
                start_age = ages[i]
                end_age = ages[i + 1]

                if end_age - start_age > 1:
                    # Linear interpolation for missing ages
                    start_rate = rates[start_age]
                    end_rate = rates[end_age]

                    for age in range(start_age + 1, end_age):
                        weight = (age - start_age) / (end_age - start_age)
                        rates[age] = start_rate + weight * (end_rate - start_rate)

        return male, female


def fetch_ssa_table(
    force_refresh: bool = False,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Convenience function to fetch SSA mortality tables.

    Args:
        force_refresh: Force download even if cache exists

    Returns:
        Tuple of (male_rates, female_rates) dictionaries
    """
    fetcher = SSATableFetcher()
    return fetcher.fetch(force_refresh)
