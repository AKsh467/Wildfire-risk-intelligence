"""Basic setup verification tests compatible with unittest and pytest."""

import unittest
from src.config import GEO_BOUNDS_INDIA, TEMPORAL_SCOPE, VIIRS_SPEC
from src.utils.logger import get_logger


class TestBasicSetup(unittest.TestCase):
    """Test suite for base configuration and logging."""

    def test_geographic_bounds(self):
        """Verify India bounding box geometry sanity."""
        self.assertLess(GEO_BOUNDS_INDIA["lat_min"], GEO_BOUNDS_INDIA["lat_max"])
        self.assertLess(GEO_BOUNDS_INDIA["lon_min"], GEO_BOUNDS_INDIA["lon_max"])
        self.assertGreaterEqual(GEO_BOUNDS_INDIA["lat_min"], 0.0)
        self.assertGreaterEqual(GEO_BOUNDS_INDIA["lon_min"], 60.0)

    def test_temporal_scope(self):
        """Verify temporal bounds consistency."""
        self.assertEqual(TEMPORAL_SCOPE["start_year"], 2020)
        self.assertEqual(TEMPORAL_SCOPE["end_year"], 2025)
        self.assertLessEqual(TEMPORAL_SCOPE["start_year"], TEMPORAL_SCOPE["end_year"])

    def test_logger_initialization(self):
        """Verify logger creation and naming."""
        logger = get_logger("test_logger")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")


if __name__ == "__main__":
    unittest.main()
