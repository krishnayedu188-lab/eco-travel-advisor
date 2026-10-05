"""
Unit tests for the Eco-Travel Advisor custom actions.
Run from the project folder with:  python -m pytest -v

External APIs (Climatiq) are mocked, so the tests are fast, free and repeatable:
no internet needed and no API quota used.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher

import actions.actions as a


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_tracker(slots=None, requested_slot=None, events=None,
                 active_loop="trip_form", latest_message=None):
    """Build a fake conversation state for testing an action."""
    slots = dict(slots or {})
    slots["requested_slot"] = requested_slot
    return Tracker(
        sender_id="test-user",
        slots=slots,
        latest_message=latest_message or {"intent": {"name": "inform"}, "entities": []},
        events=events or [],
        paused=False,
        followup_action=None,
        active_loop={"name": active_loop} if active_loop else {},
        latest_action_name=None,
    )


def fake_response(status=200, payload=None):
    """A pretend HTTP response, so no real API call is made."""
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload or {}
    response.text = json.dumps(payload or {})
    return response


def not_understood():
    return {"event": "user", "text": "blorp", "parse_data": {"intent": {"name": "out_of_scope"}}}


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------
def test_carbon_band_uses_grams_per_km():
    assert a.carbon_band(30, 1000) == "green"   # 30 g/km
    assert a.carbon_band(100, 1000) == "amber"  # 100 g/km
    assert a.carbon_band(200, 1000) == "red"    # 200 g/km


@pytest.mark.parametrize("text, expected", [
    ("Vienna", ("Vienna", False)),
    ("I'm travelling from Hamburg", ("Hamburg", False)),
    ("wien", ("Vienna", False)),            # local-language name
    ("berlinn", ("Berlin", True)),          # typo correction
    ("next week", None),                    # not a city
    ("india", None),                        # outside the supported area
])
def test_find_known_city(text, expected):
    assert a.find_known_city(text) == expected


@pytest.mark.parametrize("dates, nights", [
    ("12-15 November", 3),
    ("3 to 10 march", 7),
    ("next weekend", 2),
    ("next week", 3),   # unknown length -> default, stated to the user
])
def test_estimate_nights(dates, nights):
    assert a.estimate_nights(dates) == nights


def test_ranking_changes_with_sustainability_preference():
    options = [
        {"mode": "rail", "co2": 40, "price": 150},   # low carbon, expensive
        {"mode": "air", "co2": 200, "price": 90},    # high carbon, cheap
    ]
    high = a.rank_by_score([dict(o) for o in options], "co2", "price", "high")
    low = a.rank_by_score([dict(o) for o in options], "co2", "price", "low")
    assert high[0]["mode"] == "rail"   # 'High' sustainability puts the train first
    assert low[0]["mode"] == "air"     # 'Low' (price first) puts the cheap flight first


# ---------------------------------------------------------------------------
# Climatiq API (mocked) and fallback
# ---------------------------------------------------------------------------
def test_climatiq_success(monkeypatch):
    monkeypatch.setattr(a, "CLIMATIQ_KEY", "test-key")
    payload = {"co2e": 42.0, "emission_factor": {"source": "ADEME", "year": 2020}}
    with patch.object(a.requests, "post", return_value=fake_response(200, payload)) as post:
        result = a.climatiq_estimate("rail", "Berlin", "Vienna")
    assert result["co2e_kg"] == 42.0
    assert "Climatiq API" in result["source"]
    assert post.call_args.kwargs["timeout"] == a.API_TIMEOUT   # latency guard is used


def test_climatiq_forbidden_returns_none(monkeypatch):
    monkeypatch.setattr(a, "CLIMATIQ_KEY", "test-key")
    with patch.object(a.requests, "post", return_value=fake_response(403, {"error": "forbidden"})):
        assert a.climatiq_estimate("rail", "Berlin", "Vienna") is None


def test_climatiq_timeout_falls_back_to_offline_estimate(monkeypatch):
    monkeypatch.setattr(a, "CLIMATIQ_KEY", "test-key")
    with patch.object(a.requests, "post", side_effect=requests.Timeout):
        result = a.estimate_emissions("rail", "Berlin", "Vienna")
    assert result is not None
    assert "offline" in result["source"]


def test_no_api_key_uses_offline_estimate(monkeypatch):
    monkeypatch.setattr(a, "CLIMATIQ_KEY", "")
    result = a.estimate_emissions("air", "Berlin", "Vienna")
    assert "offline" in result["source"]


def test_calculate_carbon_sorts_lowest_first(monkeypatch):
    monkeypatch.setattr(a, "CLIMATIQ_KEY", "")
    dispatcher = CollectingDispatcher()
    tracker = make_tracker(slots={"origin": "Berlin", "destination": "Vienna"}, active_loop=None)
    a.ActionCalculateCarbon().run(dispatcher, tracker, {})
    cards = next(m["custom"] for m in dispatcher.messages if m.get("custom"))
    assert cards["options"][0]["mode"] == "bus"
    assert {o["band"] for o in cards["options"]} <= {"green", "amber", "red"}


# ---------------------------------------------------------------------------
# Form validation (regression tests for bugs found in manual testing)
# ---------------------------------------------------------------------------
def test_unknown_city_explains_scope_and_offers_human():
    dispatcher = CollectingDispatcher()
    tracker = make_tracker(requested_slot="origin", slots={"destination": "Amsterdam"})
    result = a.ValidateTripForm().validate_origin("next week", dispatcher, tracker, {})
    assert result == {"origin": None}
    message = dispatcher.messages[0]
    assert "don't have travel data" in message["text"]
    assert any(b["payload"] == "/request_human" for b in message["buttons"])


def test_destination_not_overwritten_when_answering_origin():
    events = [{"event": "slot", "name": "destination", "value": "Amsterdam"},
              {"event": "user", "text": "vienna"}]
    tracker = make_tracker(requested_slot="origin", events=events)
    result = a.ValidateTripForm().validate_destination("Vienna", CollectingDispatcher(), tracker, {})
    assert result == {"destination": "Amsterdam"}


def test_origin_same_as_destination_is_rejected():
    events = [{"event": "slot", "name": "destination", "value": "Vienna"},
              {"event": "user", "text": "vienna"}]
    dispatcher = CollectingDispatcher()
    tracker = make_tracker(requested_slot="origin", events=events)
    result = a.ValidateTripForm().validate_origin("vienna", dispatcher, tracker, {})
    assert result == {"origin": None}
    assert "same as your destination" in dispatcher.messages[0]["text"]


@pytest.mark.parametrize("value, expected", [
    ("about 600 euros", {"budget": "600"}),
    ("13", {"budget": None}),            # too low -> asked again
    ("no idea", {"budget": None}),       # no number -> asked again
    (None, {"budget": None}),            # cleared by the reset action -> silent
])
def test_budget_validation(value, expected):
    tracker = make_tracker(requested_slot="budget")
    assert a.ValidateTripForm().validate_budget(value, CollectingDispatcher(), tracker, {}) == expected


# ---------------------------------------------------------------------------
# Human handover and two-stage fallback
# ---------------------------------------------------------------------------
def test_handover_packages_full_context(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "HANDOVER_DIR", str(tmp_path))
    dispatcher = CollectingDispatcher()
    tracker = make_tracker(
        slots={"destination": "Vienna", "origin": "Berlin"},
        events=[{"event": "user", "text": "human please"}],
        latest_message={"intent": {"name": "request_human"}, "entities": []},
    )
    events = a.ActionHandover().run(dispatcher, tracker, {})

    files = list(tmp_path.glob("ECO-*.json"))
    assert len(files) == 1
    package = json.loads(files[0].read_text())
    assert package["trip"]["destination"] == "Vienna"
    assert package["transcript"][0]["text"] == "human please"
    assert dispatcher.messages[-1]["custom"]["type"] == "handover"          # frontend indicator
    assert any(e.get("event") == "active_loop" and e.get("name") is None for e in events)  # form stops


def test_fallback_two_stages_without_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "HANDOVER_DIR", str(tmp_path))

    first = CollectingDispatcher()
    a.ActionDefaultFallback().run(first, make_tracker(events=[not_understood()], active_loop=None), {})
    assert "didn't quite get that" in first.messages[0]["text"]
    assert list(tmp_path.glob("ECO-*.json")) == []                # stage 1: no handover yet

    second = CollectingDispatcher()
    a.ActionDefaultFallback().run(second, make_tracker(events=[not_understood(), not_understood()],
                                                       active_loop=None), {})
    assert any("human travel advisor" in (m.get("text") or "") for m in second.messages)
    assert len(list(tmp_path.glob("ECO-*.json"))) == 1            # stage 2: exactly ONE ticket