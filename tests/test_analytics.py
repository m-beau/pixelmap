import sqlite3
from types import SimpleNamespace

import pytest

from pixelmap.analytics import (
    _connect,
    build_request_context,
    build_visitor_key,
    get_total_visits,
    get_unique_visitors,
    get_visitor_summary,
    get_weekly_unique_visitors,
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


_VISIT_REQUEST = SimpleNamespace(
    headers={"User-Agent": "pytest-agent"},
    remote_ip="198.51.100.12",
    path="/",
)
_INSERT_SUMMARY = (
    "INSERT INTO visitor_summary (visitor_key, ip_address, first_seen_at, last_seen_at, visit_count) "
    "VALUES ('k', '198.51.100.12', 't', 't', 1)"
)


@pytest.fixture
def tracked_connections(monkeypatch):
    """Record every sqlite3 connection the analytics module opens."""
    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr("pixelmap.analytics.sqlite3.connect", tracking_connect)
    return opened


def _assert_all_closed(connections):
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda db: initialize_database(db), id="initialize_database"),
        pytest.param(
            lambda db: record_visit("visitor-1", build_request_context(_VISIT_REQUEST, "session-1"), db_path=db),
            id="record_visit",
        ),
        pytest.param(
            lambda db: record_session_end("visitor-1", "198.51.100.12", db_path=db),
            id="record_session_end",
        ),
        pytest.param(lambda db: get_total_visits(db), id="get_total_visits"),
        pytest.param(lambda db: get_unique_visitors(db), id="get_unique_visitors"),
        pytest.param(lambda db: get_visitor_summary("cookie:visitor-1", db), id="get_visitor_summary"),
        pytest.param(lambda db: get_weekly_unique_visitors(db), id="get_weekly_unique_visitors"),
    ],
)
def test_helpers_close_every_connection(tmp_path, tracked_connections, call):
    # sqlite3's own connection context manager commits/rolls back but never
    # closes, so each helper must close the connections it opens.
    call(tmp_path / "visitor_analytics.sqlite3")

    _assert_all_closed(tracked_connections)


def test_connect_commits_on_success(tmp_path, tracked_connections):
    db_path = initialize_database(tmp_path / "visitor_analytics.sqlite3")

    with _connect(db_path) as connection:
        connection.execute(_INSERT_SUMMARY)

    assert get_visitor_summary("k", db_path) is not None
    _assert_all_closed(tracked_connections)


def test_connect_rolls_back_and_closes_on_error(tmp_path, tracked_connections):
    db_path = initialize_database(tmp_path / "visitor_analytics.sqlite3")

    with pytest.raises(RuntimeError, match="boom"), _connect(db_path) as connection:
        connection.execute(_INSERT_SUMMARY)
        raise RuntimeError("boom")

    assert get_visitor_summary("k", db_path) is None
    _assert_all_closed(tracked_connections)


def _visit(db_path, visitor_id, visited_at, user_agent="Mozilla/5.0 (Macintosh)", ip="198.51.100.12"):
    request = SimpleNamespace(headers={"User-Agent": user_agent}, remote_ip=ip, path="/app")
    record_visit(
        visitor_id,
        build_request_context(request, session_id="session"),
        db_path=db_path,
        visited_at=visited_at,
    )


def test_weekly_unique_visitors_bins_monday_to_sunday_in_utc(tmp_path):
    db_path = tmp_path / "visitor_analytics.sqlite3"
    # 2026-03-23 is a Monday; 2026-03-29 the Sunday that closes its week.
    _visit(db_path, "visitor-1", "2026-03-23T00:00:00+00:00")
    _visit(db_path, "visitor-1", "2026-03-25T12:00:00+00:00")  # same visitor, same week
    _visit(db_path, "visitor-2", "2026-03-29T23:59:59+00:00")
    _visit(db_path, "visitor-1", "2026-03-30T00:00:00+00:00")  # next Monday: counted again
    # Sunday evening in New York is already Monday in UTC.
    _visit(db_path, "visitor-3", "2026-04-05T20:30:00-04:00")

    assert get_weekly_unique_visitors(db_path) == [
        {"week_start": "2026-03-23", "unique_visitors": 2},
        {"week_start": "2026-03-30", "unique_visitors": 1},
        {"week_start": "2026-04-06", "unique_visitors": 1},
    ]


def test_weekly_unique_visitors_counts_automated_browsers(tmp_path):
    db_path = tmp_path / "visitor_analytics.sqlite3"
    week = "2026-03-23T10:00:00+00:00"
    _visit(db_path, "human", week)
    # Each headless hit arrives with a fresh cookie ID, and each is a visitor.
    for index, user_agent in enumerate(
        [
            "Mozilla/5.0 (X11; Linux x86_64) HeadlessChrome/117.0.5938.132 Safari/537.36",
            "Mozilla/5.0 AppleWebKit/537.36 (compatible; bingbot/2.0) Chrome/116.0",
        ]
    ):
        _visit(db_path, f"automated-{index}", week, user_agent=user_agent)

    assert get_weekly_unique_visitors(db_path) == [{"week_start": "2026-03-23", "unique_visitors": 3}]


def test_weekly_unique_visitors_keys_visitors_like_the_summary_table(tmp_path):
    db_path = tmp_path / "visitor_analytics.sqlite3"
    week = "2026-03-23T10:00:00+00:00"
    # No cookie: keyed by IP, so two hits from one address are one visitor.
    _visit(db_path, None, week, user_agent=None, ip="203.0.113.5")
    _visit(db_path, None, week, user_agent=None, ip="203.0.113.5")
    _visit(db_path, None, week, user_agent=None, ip="203.0.113.6")
    _visit(db_path, "visitor-1", week, ip="203.0.113.5")

    assert get_weekly_unique_visitors(db_path) == [{"week_start": "2026-03-23", "unique_visitors": 3}]
    assert get_unique_visitors(db_path) == 3


def test_weekly_unique_visitors_of_an_empty_database(tmp_path):
    assert get_weekly_unique_visitors(tmp_path / "visitor_analytics.sqlite3") == []
