"""Unit tests for NILM Telemetry Web Server and Engine.

Verifies HTTP endpoints, telemetry streaming data formats, transport state machine,
and diagnostic calculators.
"""

import asyncio
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from app.server import TelemetryEngine, create_app


class TestTelemetryEngine:
    """Tests for TelemetryEngine state machine and calculations."""

    def test_engine_initialization(self) -> None:
        engine = TelemetryEngine()
        assert engine.active_feeder_id in engine.feeders
        assert engine.is_playing is True
        assert engine.speed == 1.0
        assert engine.cursor_idx == 0

    def test_transport_controls(self) -> None:
        engine = TelemetryEngine()
        engine.pause()
        assert engine.is_playing is False
        engine.play()
        assert engine.is_playing is True
        engine.set_speed(5.0)
        assert engine.speed == 5.0
        engine.seek_percent(50.0)
        assert engine.cursor_idx > 0
        engine.reset()
        assert engine.cursor_idx == 0

    def test_telemetry_packet_structure(self) -> None:
        engine = TelemetryEngine()
        packet = engine.get_telemetry_packet()
        assert "feeder" in packet
        assert "transport" in packet
        assert "electrical" in packet
        assert "appliances" in packet
        assert "telemetry" in packet
        assert "oscilloscope" in packet

        # Verify appliances
        for app_name in ["refrigerator", "dishwasher", "microwave", "washing_machine"]:
            assert app_name in packet["appliances"]
            app = packet["appliances"][app_name]
            assert "watts" in app
            assert "active" in app
            assert "prob" in app
            assert "confidence_pct" in app
            assert "substate" in app

        # Verify oscilloscope buffer length
        assert len(packet["oscilloscope"]["mains"]) == engine.window_size
        assert len(packet["oscilloscope"]["fridge"]) == engine.window_size

    def test_usage_cost_computation(self) -> None:
        engine = TelemetryEngine()
        res = engine.compute_usage_and_cost(tariff=0.20)
        assert res["tariff"] == 0.20
        assert res["daily_total_kwh"] > 0
        assert res["daily_total_cost"] > 0
        assert len(res["appliances"]) >= 4

    def test_signatures_catalog(self) -> None:
        engine = TelemetryEngine()
        sigs = engine.get_appliance_signatures()
        assert "signatures" in sigs
        assert len(sigs["signatures"]) == 4

    def test_diagnostics_report(self) -> None:
        engine = TelemetryEngine()
        diag = engine.get_diagnostics_report()
        assert "model_architecture" in diag
        assert diag["model_architecture"]["total_parameters"] == 526347
        assert diag["model_architecture"]["model_weights_fp32"] == "2.01 MB"
        assert diag["model_architecture"]["checkpoint_file_size"] == "6.08 MB"
        assert "loho_cv_benchmark" in diag
        assert diag["loho_cv_benchmark"]["appliances"][0]["f1_score"] == 0.4697


class TestServerEndpoints(AioHTTPTestCase):
    """Integration tests for HTTP routes and responses."""

    async def get_application(self) -> web.Application:
        return create_app()

    @unittest_run_loop
    async def test_get_index(self) -> None:
        resp = await self.client.request("GET", "/")
        assert resp.status == 200
        text = await resp.text()
        assert "NILM_CORE" in text

    @unittest_run_loop
    async def test_api_status(self) -> None:
        resp = await self.client.request("GET", "/api/status")
        assert resp.status == 200
        data = await resp.json()
        assert "electrical" in data
        assert "appliances" in data

    @unittest_run_loop
    async def test_api_feeders(self) -> None:
        resp = await self.client.request("GET", "/api/feeders")
        assert resp.status == 200
        data = await resp.json()
        assert "feeders" in data
        assert len(data["feeders"]) > 0

    @unittest_run_loop
    async def test_api_control(self) -> None:
        # Pause
        resp = await self.client.request("POST", "/api/control", json={"action": "pause"})
        assert resp.status == 200
        data = await resp.json()
        assert data["is_playing"] is False

        # Play
        resp = await self.client.request("POST", "/api/control", json={"action": "play"})
        assert resp.status == 200
        data = await resp.json()
        assert data["is_playing"] is True

        # Speed
        resp = await self.client.request("POST", "/api/control", json={"action": "speed", "speed": 5.0})
        assert resp.status == 200
        data = await resp.json()
        assert data["speed"] == 5.0

    @unittest_run_loop
    async def test_api_usage_cost(self) -> None:
        resp = await self.client.request("GET", "/api/usage-cost?tariff=0.18")
        assert resp.status == 200
        data = await resp.json()
        assert data["tariff"] == 0.18

    @unittest_run_loop
    async def test_api_signatures(self) -> None:
        resp = await self.client.request("GET", "/api/signatures")
        assert resp.status == 200
        data = await resp.json()
        assert "signatures" in data

    @unittest_run_loop
    async def test_api_sensor_feeds(self) -> None:
        resp = await self.client.request("GET", "/api/sensor-feeds")
        assert resp.status == 200
        data = await resp.json()
        assert "adc_pipeline" in data
        assert "power_quality" in data

    @unittest_run_loop
    async def test_api_diagnostics(self) -> None:
        resp = await self.client.request("GET", "/api/diagnostics")
        assert resp.status == 200
        data = await resp.json()
        assert "loho_cv_benchmark" in data
        apps = {a["appliance"]: a for a in data["loho_cv_benchmark"]["appliances"]}
        assert "Dishwasher" in apps
        # Confirm dishwasher is calibrated, not uncalibrated 0.1779
        assert apps["Dishwasher"]["f1_score"] == 0.3625
        assert apps["Dishwasher"]["status"] == "CALIBRATED_OPTIMAL"
        assert "Few-Shot" in apps["Dishwasher"]["calibration"]
        assert "Refrigerator" in apps
        assert apps["Refrigerator"]["f1_score"] == 0.4697
        if "combined_phase7_benchmark" in data:
            assert "appliances" in data["combined_phase7_benchmark"]

    @unittest_run_loop
    async def test_api_export(self) -> None:
        resp = await self.client.request("GET", "/api/export")
        assert resp.status == 200
        assert resp.content_type == "text/csv"
        text = await resp.text()
        assert "datetime,mains_w" in text
