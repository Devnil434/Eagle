"""
tests/test_zone_loader.py

Unit tests for libs/config/zone_loader.py
Covers all acceptance criteria from Issue #44.
"""

import os
import time
import pytest
from unittest.mock import patch

from libs.config.zone_loader import load_zones, ZoneConfigLoader, _validate_polygon


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_YAML = """
camera_id: cam_test
zones:
  - name: restricted_door
    polygon: [[540,200],[740,200],[740,480],[540,480]]
    alert_on_entry: true
    color_hex: "#ef4444"
  - name: safe_corridor
    polygon: [[0,0],[300,0],[300,480],[0,480]]
    alert_on_entry: false
    color_hex: "#22c55e"
"""

INVALID_POLYGON_YAML = """
camera_id: cam_test
zones:
  - name: bad_zone
    polygon: [[0,0],[1,1]]
    alert_on_entry: true
    color_hex: "#ff0000"
"""

NON_LIST_POLYGON_YAML = """
camera_id: cam_test
zones:
  - name: bad_zone
    polygon: "not_a_list"
    alert_on_entry: true
    color_hex: "#ff0000"
"""

MISSING_POLYGON_YAML = """
camera_id: cam_test
zones:
  - name: no_polygon_zone
    alert_on_entry: true
    color_hex: "#ff0000"
"""


@pytest.fixture
def valid_config_file(tmp_path):
    f = tmp_path / "zones.yaml"
    f.write_text(VALID_YAML, encoding="utf-8")
    return f


@pytest.fixture
def invalid_polygon_file(tmp_path):
    f = tmp_path / "zones_bad.yaml"
    f.write_text(INVALID_POLYGON_YAML, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# _validate_polygon
# ---------------------------------------------------------------------------

class TestValidatePolygon:
    def test_valid_polygon(self):
        _validate_polygon([[0,0],[1,0],[1,1]], "test_zone")  # no raise

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            _validate_polygon([[0,0],[1,1]], "z")

    def test_not_a_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _validate_polygon("bad", "z")

    def test_bad_point_raises(self):
        with pytest.raises(ValueError, match="pair of numbers"):
            _validate_polygon([[0,0],[1,"x"],[2,2]], "z")

    def test_point_wrong_length_raises(self):
        with pytest.raises(ValueError, match="pair of numbers"):
            _validate_polygon([[0,0,0],[1,1,1],[2,2,2]], "z")

    def test_float_coords_valid(self):
        _validate_polygon([[0.0,0.0],[1.5,0.0],[1.5,1.5]], "z")  # no raise


# ---------------------------------------------------------------------------
# load_zones
# ---------------------------------------------------------------------------

class TestLoadZones:
    def test_loads_valid_yaml(self, valid_config_file):
        config = load_zones(valid_config_file)
        assert config["camera_id"] == "cam_test"
        assert len(config["zones"]) == 2

    def test_zone_fields_present(self, valid_config_file):
        config = load_zones(valid_config_file)
        zone = config["zones"][0]
        assert zone["name"] == "restricted_door"
        assert zone["alert_on_entry"] is True
        assert zone["color_hex"] == "#ef4444"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_zones(tmp_path / "nonexistent.yaml")

    def test_invalid_polygon_raises_value_error(self, invalid_polygon_file):
        with pytest.raises(ValueError, match="at least 3 points"):
            load_zones(invalid_polygon_file)

    def test_missing_polygon_key_raises(self, tmp_path):
        f = tmp_path / "z.yaml"
        f.write_text(MISSING_POLYGON_YAML, encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'polygon'"):
            load_zones(f)

    def test_non_list_polygon_raises(self, tmp_path):
        f = tmp_path / "z.yaml"
        f.write_text(NON_LIST_POLYGON_YAML, encoding="utf-8")
        with pytest.raises(ValueError, match="must be a list"):
            load_zones(f)

    def test_zones_config_path_env_var(self, valid_config_file):
        with patch.dict(os.environ, {"ZONES_CONFIG_PATH": str(valid_config_file)}):
            config = load_zones()
            assert len(config["zones"]) == 2


# ---------------------------------------------------------------------------
# ZoneConfigLoader
# ---------------------------------------------------------------------------

class TestZoneConfigLoader:
    def test_get_zones_returns_list(self, valid_config_file):
        loader = ZoneConfigLoader(config_path=valid_config_file)
        zones = loader.get_zones()
        assert isinstance(zones, list)
        assert len(zones) == 2

    def test_get_camera_id(self, valid_config_file):
        loader = ZoneConfigLoader(config_path=valid_config_file)
        assert loader.get_camera_id() == "cam_test"

    def test_start_stop(self, valid_config_file):
        loader = ZoneConfigLoader(config_path=valid_config_file, reload_interval=1)
        loader.start()
        assert loader._thread.is_alive()
        loader.stop()
        loader._thread.join(timeout=3)
        assert not loader._thread.is_alive()

    def test_hot_reload_picks_up_changes(self, tmp_path):
        config_file = tmp_path / "zones.yaml"
        config_file.write_text(VALID_YAML, encoding="utf-8")

        loader = ZoneConfigLoader(config_path=config_file, reload_interval=1)
        assert len(loader.get_zones()) == 2

        # Write updated config with 1 zone
        updated = """
camera_id: cam_updated
zones:
  - name: only_zone
    polygon: [[0,0],[100,0],[100,100],[0,100]]
    alert_on_entry: false
    color_hex: "#00ff00"
"""
        config_file.write_text(updated, encoding="utf-8")
        loader.start()
        time.sleep(2.5)  # wait for at least one reload cycle
        loader.stop()

        assert len(loader.get_zones()) == 1
        assert loader.get_camera_id() == "cam_updated"

    def test_missing_file_on_reload_keeps_previous_config(self, tmp_path):
        config_file = tmp_path / "zones.yaml"
        config_file.write_text(VALID_YAML, encoding="utf-8")

        loader = ZoneConfigLoader(config_path=config_file, reload_interval=1)
        assert len(loader.get_zones()) == 2

        config_file.unlink()  # delete the file
        loader._reload()     # trigger reload manually

        # Should still return previous zones
        assert len(loader.get_zones()) == 2
