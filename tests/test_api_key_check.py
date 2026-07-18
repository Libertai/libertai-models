import json

from src.api_keys import KeysManager, apply_key_payload, check_api_key


def _reset(keys=(), invalid=None):
    KeysManager().reset_keys(set(keys), dict(invalid or {}))


def test_apply_payload_with_invalid_map():
    count = apply_key_payload({"keys": ["a"], "invalid_keys": {"b": {"reason": "expired", "message": "m"}}})
    assert count == 1
    assert KeysManager().key_exists("a")
    assert KeysManager().invalid_keys == {"b": {"reason": "expired", "message": "m"}}


def test_apply_payload_without_invalid_map_clears_stale_entries():
    _reset(invalid={"old": {"reason": "expired", "message": "m"}})
    apply_key_payload({"keys": ["a"]})
    assert KeysManager().invalid_keys == {}


def test_check_valid_key():
    _reset(keys=["good"])
    assert check_api_key("good") is None


def test_check_blocked_key_403_openai_shape():
    _reset(invalid={"blocked": {"reason": "no_credits", "message": "No credits."}})
    resp = check_api_key("blocked")
    assert resp.status_code == 403
    body = json.loads(resp.body)
    assert body == {"error": {"message": "No credits.", "type": "invalid_request_error", "code": "no_credits"}}


def test_check_unknown_key_401_legacy_shape():
    _reset()
    resp = check_api_key("nope")
    assert resp.status_code == 401
    assert json.loads(resp.body) == {"detail": "Invalid API key"}
