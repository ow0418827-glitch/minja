"""Offline audits of prepared/persistent original-image caching and episodes."""
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import quote

from PIL import Image
import pytest
import requests

from src.agent import minja
from src.memory import multimodal_memory as images


ASIN = "B000000001"
URL = "https://images.example.invalid/original.png"


def png_bytes(color="blue"):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def observed(url=URL, asin=ASIN, **extra):
    return {"ASIN": asin, "Title": "正常商品", "PageType": "item", "SourceURL": url, **extra}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def prepared_cache(root, *, body=None, overrides=None):
    body = png_bytes() if body is None else body
    asset = root / "assets" / "甲 原图.png"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(body)
    record = {"asin": ASIN, "url": URL, "path": "assets/甲 原图.png",
              "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "role": "original_product"}
    record.update(overrides or {})
    record_path = root / "image_records" / (ASIN + ".json")
    write_json(record_path, record)
    return asset, record_path, body


def forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Offline cache audit attempted a network request")
    monkeypatch.setattr(images.requests, "get", fail)


class Response:
    def __init__(self, body, *, headers=None):
        self.body = body
        self.headers = headers or {"Content-Type": "image/png"}
        self.closed = False
        self.stream_reads = 0

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        self.stream_reads += 1
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start:start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


@pytest.mark.parametrize("local_alias", [False, True])
def test_prepared_original_is_copied_offline_and_validated(tmp_path, monkeypatch, local_alias):
    root, memory = tmp_path / "prepared", tmp_path / "bank" / "memory.json"
    original, record_path, body = prepared_cache(root)
    page_url = "http://127.0.0.1:3012/minja-assets/" + quote("甲 原图.png") if local_alias else URL
    forbid_network(monkeypatch)
    saved = images.snapshot_product_image(observed(page_url), memory, image_cache_root=root)
    local, actual, metadata = images.validated_image_bytes(saved, memory)
    assert actual == body
    assert local != original
    assert local.is_relative_to(memory.parent)
    assert saved["SourceURL"] == page_url
    assert saved["OriginalURL"] == URL
    assert saved["OriginalImageCacheRecord"] == str(record_path)
    assert saved["ImageCacheSource"] == "prepared"
    assert metadata["SHA256"] == hashlib.sha256(body).hexdigest()
    assert not Path(saved["LocalPath"]).is_absolute()
    assert not (root / "original_image_cache").exists()


def test_unknown_remote_retries_with_bounds_then_persists_for_offline_next_bank(tmp_path, monkeypatch):
    root, first_memory = tmp_path / "cache", tmp_path / "first" / "memory.json"
    body, calls, response = png_bytes(), [], Response(png_bytes())
    def download(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise requests.Timeout("first bounded attempt")
        return response
    monkeypatch.setattr(images.requests, "get", download)
    first = images.snapshot_product_image(observed(), first_memory, image_cache_root=root,
                                          timeout=(0.01, 0.02), max_attempts=2)
    assert len(calls) == 2
    assert all(url == URL and options["timeout"] == (0.01, 0.02) and options["stream"] is True
               for url, options in calls)
    assert response.closed is True
    assert first["ImageCacheSource"] == "download"
    assert images.validated_image_bytes(first, first_memory)[1] == body
    cache_record = Path(first["OriginalImageCacheRecord"])
    record = json.loads(cache_record.read_text(encoding="utf-8"))
    assert record["role"] == "original_product" and record["asin"] == ASIN and record["url"] == URL
    assert cache_record.is_relative_to(root / "original_image_cache")
    forbid_network(monkeypatch)
    second_memory = tmp_path / "second" / "memory.json"
    second = images.snapshot_product_image(observed(), second_memory, image_cache_root=root)
    assert second["ImageCacheSource"] == "persistent"
    assert second["SHA256"] == first["SHA256"]
    assert images.validated_image_bytes(second, second_memory)[1] == body
    assert (second_memory.parent / second["LocalPath"]).is_file()


def test_download_failure_stops_at_configured_attempt_count(tmp_path, monkeypatch):
    calls = []
    def unavailable(url, **kwargs):
        calls.append(kwargs)
        raise requests.ConnectionError("offline fixture")
    monkeypatch.setattr(images.requests, "get", unavailable)
    with pytest.raises(requests.ConnectionError):
        images.snapshot_product_image(observed(), tmp_path / "bank" / "memory.json",
                                      image_cache_root=tmp_path / "cache", timeout=(0.01, 0.02), max_attempts=3)
    assert len(calls) == 3
    assert all(call["timeout"] == (0.01, 0.02) for call in calls)
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("declared", [False, True])
def test_download_size_limit_prevents_unbounded_asset_cache(tmp_path, monkeypatch, declared):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 8)
    response = Response(b"0123456789", headers={"Content-Length": "10"} if declared else {})
    calls = []
    def download(url, **kwargs):
        calls.append(url)
        return response
    monkeypatch.setattr(images.requests, "get", download)
    with pytest.raises(ValueError, match="maximum download size"):
        images.snapshot_product_image(observed(), tmp_path / "bank" / "memory.json",
                                      image_cache_root=tmp_path / "cache", max_attempts=2)
    assert len(calls) == 1
    assert response.closed is True
    assert response.stream_reads == (0 if declared else 1)
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("mutation, match", [
    ({"sha256": "0" * 64}, "digest mismatch"),
    ({"sha256": None}, "SHA256 is required"),
    ({"role": "attack"}, "Attack or substituted"),
    ({"role": "adversarial_product"}, "Attack or substituted"),
    ({"DepictedASIN": "OTHER"}, "Attack or substituted"),
    ({"asin": "OTHER"}, "ASIN does not match"),
    ({"path": "../outside.png"}, "escapes its assets directory"),
])
def test_bad_prepared_original_record_is_rejected_without_fetch(tmp_path, monkeypatch, mutation, match):
    root = tmp_path / "cache"
    prepared_cache(root, overrides=mutation)
    forbid_network(monkeypatch)
    with pytest.raises(ValueError, match=match):
        images.snapshot_product_image(observed(), tmp_path / "bank" / "memory.json", image_cache_root=root)


@pytest.mark.parametrize("extra", [
    {"Role": "attack"}, {"is_adversarial": True}, {"AttackSourceASIN": "OTHER"},
    {"Role": "controlled_image_substitution"}, {"EvidenceOriginalPrimaryImage": {"ASIN": ASIN}},
])
def test_attack_input_never_reuses_original_cache(tmp_path, monkeypatch, extra):
    root = tmp_path / "cache"
    prepared_cache(root)
    forbid_network(monkeypatch)
    with pytest.raises(ValueError, match="Attack or substituted"):
        images.snapshot_product_image(observed(**extra), tmp_path / "bank" / "memory.json", image_cache_root=root)


def test_unknown_local_override_is_not_downloaded_as_original(tmp_path, monkeypatch):
    forbid_network(monkeypatch)
    with pytest.raises(ValueError, match="audited original cache record"):
        images.snapshot_product_image(observed("http://localhost:3012/minja-assets/new.png"),
                                      tmp_path / "bank" / "memory.json", image_cache_root=tmp_path / "cache")


def test_same_product_new_url_gets_separate_persistent_record(tmp_path, monkeypatch):
    root, bank = tmp_path / "cache", tmp_path / "bank" / "memory.json"
    prepared_cache(root)
    new_url, new_body, calls = URL + "?version=2", png_bytes("green"), []
    def download(url, **kwargs):
        calls.append(url)
        return Response(new_body)
    monkeypatch.setattr(images.requests, "get", download)
    saved = images.snapshot_product_image(observed(new_url), bank, image_cache_root=root)
    assert calls == [new_url]
    assert saved["OriginalURL"] == new_url
    assert saved["ImageCacheSource"] == "download"
    assert images.validated_image_bytes(saved, bank)[1] == new_body
    forbid_network(monkeypatch)
    previous = images.snapshot_product_image(observed(), bank, image_cache_root=root)
    current = images.snapshot_product_image(observed(new_url), bank, image_cache_root=root)
    assert previous["ImageCacheSource"] == "prepared" and current["ImageCacheSource"] == "persistent"
    assert previous["SHA256"] != current["SHA256"]


def fake_pages(monkeypatch, source):
    def page(**state):
        kind = state["page_type"]
        info = {"instruction": "Buy the normal product"}
        if kind == "search":
            info.update(asins=[ASIN], has_next=False)
        elif kind == "item":
            image = observed(source["url"])
            info.update(current_item_image=image, images=[copy.deepcopy(image)],
                        current_item_price=12.5, option_types={"blue": "color", "red": "color"})
        elif kind == "end":
            info["reward"] = 1.0
        return "offline page " + kind, info
    monkeypatch.setattr(minja, "webshop_text", page)


def visit_item(env, session="episode"):
    env.step(session, "reset")
    env.step(session, "search[normal product]")
    return env.step(session, "click[" + ASIN + "]")[3]["current_item_image"]


def test_episode_same_asin_url_repeated_visits_only_download_once_and_freeze_purchase(tmp_path, monkeypatch):
    source, calls = {"url": URL}, []
    fake_pages(monkeypatch, source)
    def download(url, **kwargs):
        calls.append(url)
        return Response(png_bytes())
    monkeypatch.setattr(images.requests, "get", download)
    env = minja.webshopEnv(image_snapshot_path=tmp_path / "bank" / "memory.json",
                           image_cache_root=tmp_path / "cache", image_timeout=(0.01, 0.02), image_max_attempts=1)
    first = visit_item(env)
    assert first["FirstObservedStep"] == 3 and first["ImageSnapshotCacheHit"] is False
    repeated = env.step("episode", "click[blue]")[3]["current_item_image"]
    assert repeated["ImageSnapshotCacheHit"] is True
    assert repeated["FirstObservedStep"] == 3 and repeated["Step"] == 4
    assert repeated["SHA256"] == first["SHA256"]
    env.step("episode", "click[< Prev]")
    revisited = env.step("episode", "click[" + ASIN + "]")[3]["current_item_image"]
    assert revisited["ImageSnapshotCacheHit"] is True
    assert calls == [URL]
    _, reward, done, final = env.step("episode", "click[Buy Now]")
    assert done is True and reward == 1.0
    assert final["final_asin"] == ASIN and final["final_image"]["SHA256"] == first["SHA256"]
    assert Path(final["final_image"]["LocalPath"]).is_absolute()


def test_episode_reset_rechecks_persistent_cache_and_keeps_sessions_isolated(tmp_path, monkeypatch):
    source, calls = {"url": URL}, []
    fake_pages(monkeypatch, source)
    def download(url, **kwargs):
        calls.append(url)
        return Response(png_bytes())
    monkeypatch.setattr(images.requests, "get", download)
    env = minja.webshopEnv(image_snapshot_path=tmp_path / "bank" / "memory.json", image_cache_root=tmp_path / "cache")
    first = visit_item(env, "one")
    second_session = visit_item(env, "two")
    assert second_session["ImageCacheSource"] == "persistent"
    assert second_session["ImageSnapshotCacheHit"] is False
    two_cache = copy.deepcopy(env._episode_image_snapshots["two"])
    env.step("one", "click[Buy Now]")
    assert env.sessions["one"]["final_image"]["SHA256"] == first["SHA256"]
    forbid_network(monkeypatch)
    env.step("one", "reset")
    assert env._episode_image_snapshots["one"] == {}
    assert env._episode_image_snapshots["two"] == two_cache
    assert "final_image" not in env.sessions["one"]
    env.step("one", "search[normal product]")
    restarted = env.step("one", "click[" + ASIN + "]")[3]["current_item_image"]
    assert restarted["ImageCacheSource"] == "persistent"
    assert restarted["ImageSnapshotCacheHit"] is False and restarted["FirstObservedStep"] == 3
    assert restarted["SHA256"] == first["SHA256"]
    assert calls == [URL]


def test_episode_same_asin_changed_url_is_not_an_episode_cache_hit(tmp_path, monkeypatch):
    source, calls = {"url": URL}, []
    fake_pages(monkeypatch, source)
    def download(url, **kwargs):
        calls.append(url)
        return Response(png_bytes("green" if "version=2" in url else "blue"))
    monkeypatch.setattr(images.requests, "get", download)
    env = minja.webshopEnv(image_snapshot_path=tmp_path / "bank" / "memory.json", image_cache_root=tmp_path / "cache")
    old = visit_item(env)
    source["url"] = URL + "?version=2"
    new = env.step("episode", "click[red]")[3]["current_item_image"]
    assert calls == [URL, source["url"]]
    assert new["ImageSnapshotCacheHit"] is False and new["FirstObservedStep"] == 4
    assert new["SHA256"] != old["SHA256"]


def test_episode_reuse_rejects_tampered_snapshot_without_fetch(tmp_path, monkeypatch):
    source = {"url": URL}
    fake_pages(monkeypatch, source)
    root = tmp_path / "cache"
    prepared_cache(root)
    forbid_network(monkeypatch)
    env = minja.webshopEnv(image_snapshot_path=tmp_path / "bank" / "memory.json", image_cache_root=root)
    first = visit_item(env)
    Path(first["LocalPath"]).write_bytes(png_bytes("red"))
    with pytest.raises(ValueError, match="digest mismatch"):
        env.step("episode", "click[blue]")
