from types import SimpleNamespace

from pixelmap.analytics import (
    build_request_context,
    build_visitor_key,
    get_total_visits,
    get_unique_visitors,
    get_visitor_summary,
    initialize_database,
    record_session_end,
    record_visit,
    resolve_analytics_db_path,
)


def test_resolve_analytics_db_path_from_env(monkeypatch, tmp_path):
    custom_db_path = tmp_path / "analytics" / "visitors.sqlite3"
    monkeypatch.setenv("ANALYTICS_DB_PATH", str(custom_db_path))

    assert resolve_analytics_db_path() == custom_db_path


def test_initialize_database_creates_expected_tables(tmp_path):
    db_path = tmp_path / "visitor_analytics.sqlite3"

    initialize_database(db_path)

    assert db_path.exists()
    assert get_total_visits(db_path) == 0
    assert get_unique_visitors(db_path) == 0


def test_build_request_context_prefers_forwarded_ip_when_enabled():
    request = SimpleNamespace(
        headers={
            "User-Agent": "pytest-agent",
            "X-Forwarded-For": "203.0.113.10, 10.0.0.5",
        },
        remote_ip="10.0.0.5",
        path="/",
    )

    context = build_request_context(request, session_id="session-1", trust_proxy_headers=True)

    assert context.session_id == "session-1"
    assert context.ip_address == "203.0.113.10"
    assert context.forwarded_for == "203.0.113.10, 10.0.0.5"
    assert context.user_agent == "pytest-agent"
    assert context.path == "/"


def test_build_visitor_key_falls_back_to_ip():
    assert build_visitor_key(None, "198.51.100.12") == "ip:198.51.100.12"


def test_record_visit_tracks_total_and_unique_visitors(tmp_path):
    db_path = tmp_path / "visitor_analytics.sqlite3"
    first_request = SimpleNamespace(
        headers={"User-Agent": "pytest-agent"},
        remote_ip="198.51.100.12",
        path="/",
    )
    second_request = SimpleNamespace(
        headers={"User-Agent": "pytest-agent"},
        remote_ip="198.51.100.12",
        path="/download",
    )

    first_context = build_request_context(first_request, session_id="session-1")
    second_context = build_request_context(second_request, session_id="session-2")

    record_visit("visitor-1", first_context, db_path=db_path, visited_at="2026-03-19T18:00:00+00:00")
    record_visit("visitor-1", second_context, db_path=db_path, visited_at="2026-03-19T18:05:00+00:00")
    record_visit("visitor-2", second_context, db_path=db_path, visited_at="2026-03-19T18:10:00+00:00")

    assert get_total_visits(db_path) == 3
    assert get_unique_visitors(db_path) == 2

    visitor_summary = get_visitor_summary("cookie:visitor-1", db_path=db_path)
    assert visitor_summary is not None
    assert visitor_summary["visit_count"] == 2
    assert visitor_summary["first_seen_at"] == "2026-03-19T18:00:00+00:00"
    assert visitor_summary["last_seen_at"] == "2026-03-19T18:05:00+00:00"


def test_record_session_end_updates_last_seen(tmp_path):
    db_path = tmp_path / "visitor_analytics.sqlite3"
    request = SimpleNamespace(headers={}, remote_ip="203.0.113.5", path="/")
    context = build_request_context(request, session_id="session-1")

    record_visit("visitor-1", context, db_path=db_path, visited_at="2026-03-19T18:00:00+00:00")
    record_session_end(
        "visitor-1",
        "203.0.113.5",
        db_path=db_path,
        ended_at="2026-03-19T18:20:00+00:00",
    )

    visitor_summary = get_visitor_summary("cookie:visitor-1", db_path=db_path)
    assert visitor_summary is not None
    assert visitor_summary["last_seen_at"] == "2026-03-19T18:20:00+00:00"
