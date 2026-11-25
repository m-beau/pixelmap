"""Tests for the /usage-stats endpoint and the script that draws its graph."""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application

from pixelmap import usage_stats
from pixelmap.analytics import build_request_context, record_visit

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "entrypoint.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "usage-stats.yml"
SCRIPT = ROOT / "scripts" / "update_usage_graph.py"

# What a visit stores besides its week: none of it may reach the endpoint.
VISITOR_ID = "3f2b8c1e-visitor-cookie"
VISITOR_IP = "198.51.100.12"
VISITOR_USER_AGENT = "Mozilla/5.0 (Macintosh; pytest)"


def _load_script():
    spec = importlib.util.spec_from_file_location("update_usage_graph", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


update_usage_graph = _load_script()


@pytest.fixture
def analytics_db(monkeypatch, tmp_path):
    db_path = tmp_path / "visitor_analytics.sqlite3"
    monkeypatch.setenv("ANALYTICS_DB_PATH", str(db_path))
    monkeypatch.setattr(usage_stats, "_cached_payload", None)
    request = SimpleNamespace(
        headers={"User-Agent": VISITOR_USER_AGENT}, remote_ip=VISITOR_IP, path="/app"
    )
    record_visit(
        VISITOR_ID,
        build_request_context(request, session_id="session-1"),
        db_path=db_path,
        visited_at="2026-03-25T12:00:00+00:00",
    )
    return db_path


def test_entrypoint_serves_the_endpoint_the_workflow_fetches():
    # The option on the command line, not its mention in the comment above it.
    plugin = re.search(r"^\s+--plugins\s+(\S+)", ENTRYPOINT.read_text(), re.MULTILINE)
    assert plugin, "entrypoint.sh must pass --plugins to panel serve"
    assert plugin.group(1) == usage_stats.__name__
    # `panel serve --plugins` reads the module's ROUTES.
    assert [route[0] for route in usage_stats.ROUTES] == [usage_stats.ENDPOINT]

    url = re.search(r"USAGE_STATS_URL:\s*(\S+)", WORKFLOW.read_text())
    assert url and url.group(1).endswith(usage_stats.ENDPOINT)
    assert update_usage_graph.DEFAULT_URL == url.group(1)


def test_payload_is_cached_between_requests(analytics_db):
    first = usage_stats.get_cached_payload()
    record_visit(
        "another-visitor",
        build_request_context(SimpleNamespace(headers={}, remote_ip=VISITOR_IP, path="/app")),
        db_path=analytics_db,
        visited_at="2026-03-26T12:00:00+00:00",
    )

    assert usage_stats.get_cached_payload() is first
    assert first["weeks"] == [{"week_start": "2026-03-23", "unique_visitors": 1}]


class TestUsageStatsHandler(AsyncHTTPTestCase):
    def get_app(self):
        return Application(usage_stats.ROUTES)

    @pytest.fixture(autouse=True)
    def _database(self, analytics_db):
        pass

    def test_serves_weekly_counts_and_nothing_that_identifies_a_visitor(self):
        response = self.fetch(usage_stats.ENDPOINT)

        assert response.code == 200
        assert response.headers["Content-Type"].startswith("application/json")
        payload = json.loads(response.body)
        assert payload["weeks"] == [{"week_start": "2026-03-23", "unique_visitors": 1}]
        # The script reads the endpoint's own output.
        assert update_usage_graph.parse_weeks(payload["weeks"]) == {date(2026, 3, 23): 1}
        body = response.body.decode()
        for private in (VISITOR_ID, VISITOR_IP, VISITOR_USER_AGENT):
            assert private not in body


def test_merge_keeps_remembered_weeks_and_drops_the_week_in_progress():
    history = {date(2026, 3, 23): 27, date(2026, 3, 30): 30}
    fetched = {date(2026, 3, 30): 32, date(2026, 4, 6): 27, date(2026, 4, 13): 5}

    # Wednesday 2026-04-15: the week of April 13 is not over yet.
    merged = update_usage_graph.merge_weeks(history, fetched, today=date(2026, 4, 15))

    assert merged == {date(2026, 3, 23): 27, date(2026, 3, 30): 32, date(2026, 4, 6): 27}
    # The following Monday it is.
    assert date(2026, 4, 13) in update_usage_graph.merge_weeks(history, fetched, today=date(2026, 4, 20))


def test_parse_weeks_rejects_a_week_that_does_not_start_on_monday():
    with pytest.raises(ValueError, match="not a Monday"):
        update_usage_graph.parse_weeks([{"week_start": "2026-03-24", "unique_visitors": 3}])


def test_script_writes_the_table_and_the_graph(tmp_path):
    payload = tmp_path / "usage-stats.json"
    payload.write_text(
        json.dumps(
            {
                "weeks": [
                    {"week_start": "2026-03-23", "unique_visitors": 27},
                    {"week_start": "2026-03-30", "unique_visitors": 32},
                ]
            }
        )
    )
    out_dir = tmp_path / "out"

    assert update_usage_graph.main(["--url", payload.as_uri(), "--out-dir", str(out_dir)]) == 0

    csv_text = (out_dir / update_usage_graph.CSV_NAME).read_text()
    assert csv_text == "week_start,unique_visitors\n2026-03-23,27\n2026-03-30,32\n"
    assert (out_dir / update_usage_graph.PNG_NAME).read_bytes().startswith(b"\x89PNG")
    # A second run reads its own CSV back and changes nothing.
    assert update_usage_graph.main(["--url", payload.as_uri(), "--out-dir", str(out_dir)]) == 0
    assert (out_dir / update_usage_graph.CSV_NAME).read_text() == csv_text
