import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from src.environment.disk_catalog import DiskCatalog, build_catalog, fingerprint
from src.environment import disk_catalog, full_webshop


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def raw_product(asin, name):
    return {"asin": asin, "name": name, "full_description": "A normal item",
            "small_description": ["Useful"], "pricing": "$10.0", "customization_options": {},
            "images": ["https://example.test/original.png"], "query": "normal items",
            "category": "test", "product_category": "Test"}


def goal(asin="A000000001"):
    return {"asin": asin, "category": "test", "query": "normal items", "name": "甲商品",
            "product_category": "Test", "instruction_text": "Buy the normal item",
            "attributes": ["useful"], "price_upper": 20, "goal_options": {}, "weight": 1}


@pytest.fixture
def catalog_root(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_catalog, "PROJECT_ROOT", tmp_path)
    items, attrs, human = [tmp_path / name for name in ("items.json", "attrs.json", "human.json")]
    write(items, [raw_product("A000000001", "甲商品"), raw_product("B000000002", "乙商品"),
                  raw_product("A000000001", "重复甲商品")])
    write(attrs, {"A000000001": {"attributes": ["useful"]}, "B000000002": {"attributes": ["other"]}})
    write(human, {"A000000001": []})
    destination = tmp_path / "src" / "output" / "full"
    build_catalog(items, attrs, human, destination, tmp_path,
                  goal_builder=lambda products, prices: [goal()], search_backend="sqlite_fts5")
    return destination


def test_build_entire_array_and_keep_source_read_only(catalog_root):
    manifest = json.loads((catalog_root / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == "full_catalog"
    assert manifest["source_records_seen"] == 3
    assert manifest["products"] == 2
    assert manifest["skipped_invalid_or_duplicate"] == 1
    assert not (catalog_root / "products.jsonl").exists()
    assert manifest["source_files"]["items"]["bytes"] > 0


def test_startup_decodes_no_products_and_pagination_is_bounded(catalog_root):
    with DiskCatalog.open(catalog_root) as catalog:
        assert catalog.diagnostics()["decoded_products"] == 0
        assert len(catalog.sequence) == 2
        assert catalog.sequence[0:1][0]["asin"] == "A000000001"
        assert catalog.decoded_products == 1
        filtered = catalog.search(["<a>", "useful"])
        assert len(filtered) == 1
        assert catalog.decoded_products == 1
        assert filtered[0]["asin"] == "A000000001"
        assert catalog.decoded_products == 2
        assert catalog.sequence[-1]["asin"] == "B000000002"
        with pytest.raises(IndexError):
            catalog.sequence[-3]
        with pytest.raises(sqlite3.OperationalError):
            catalog.conn.execute("DELETE FROM products")
        assert catalog.diagnostics()["sqlite_cache_kib"] == 16384


def test_fts_natural_query_and_special_queries(catalog_root):
    with DiskCatalog.open(catalog_root) as catalog:
        assert {p["asin"] for p in catalog.search(["useful", "OR", '"'])} == {"A000000001", "B000000002"}
        assert len(catalog.search(["<c>", "test"])) == 2
        assert len(catalog.search(["<q>", "normal", "items"])) == 2
        assert len(catalog.search(["<r>"])) == 2
        assert catalog.search([]) == []
        with pytest.raises(KeyError):
            catalog.product("missing")


def test_manifest_mismatch_is_rejected(catalog_root):
    index = json.loads((catalog_root / "index_manifest.json").read_text(encoding="utf-8"))
    index["bundle_id"] = "wrong"
    write(catalog_root / "index_manifest.json", index)
    with pytest.raises(ValueError, match="bundle mismatch"):
        DiskCatalog.open(catalog_root)


def test_cannot_write_catalog_outside_current_project_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_catalog, "PROJECT_ROOT", tmp_path / "minja")
    with pytest.raises(ValueError, match="output directory"):
        build_catalog("missing", "missing", "missing", tmp_path / "other_source", tmp_path)


def test_old_rap_sqlite_schema_can_be_opened_without_raw_json(catalog_root):
    conn = sqlite3.connect(catalog_root / "catalog.sqlite")
    conn.execute("DROP INDEX product_category_lookup")
    conn.execute("DROP INDEX product_query_lookup")
    conn.execute("ALTER TABLE products DROP COLUMN category")
    conn.execute("ALTER TABLE products DROP COLUMN query")
    conn.commit()
    conn.close()
    with DiskCatalog.open(catalog_root) as catalog:
        assert len(catalog.search(["<c>", "test"])) == 2
        assert len(catalog.search(["<q>", "normal", "items"])) == 2


def test_real_goal_registry_reset_and_dynamic_image_overlay(catalog_root, tmp_path, monkeypatch):
    from flask import Flask, jsonify
    source = tmp_path / "source"
    (source / "web_agent_site").mkdir(parents=True)
    (source / "web_agent_site" / "app.py").write_text("# fixture", encoding="utf-8")
    site = SimpleNamespace(app=Flask("disk-test"), user_sessions={})

    @site.app.route("/<session_id>")
    def index(session_id):
        # The initialized searcher prevents the original lazy all-product load.
        assert site.search_engine is not None
        return jsonify(site.user_sessions[session_id])

    @site.app.route("/rap/page")
    def rap_page():
        from flask import request
        return jsonify(site.user_sessions[request.args["session_id"]])

    monkeypatch.setattr(full_webshop.importlib, "import_module", lambda name: site)
    from src.environment import local_webshop
    monkeypatch.setattr(local_webshop, "configure_java", lambda: None)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "adversarial.png").write_bytes(b"fixture")
    registry, overlays = tmp_path / "registry.json", tmp_path / "overlays.json"
    write(registry, {"custom_1": goal()})
    write(overlays, {"A000000001": {"path": "adversarial.png"}})
    adapted, metadata = full_webshop.make_site(catalog_root, source, task_registry=registry,
                                               image_overrides=overlays, asset_root=assets)
    client = adapted.app.test_client()
    assert client.get("/custom_1").json["goal"]["asin"] == "A000000001"
    adapted.user_sessions["custom_1"]["done"] = True
    assert client.get("/custom_1").json["done"] is False
    adapted.user_sessions.clear()
    assert client.get("/rap/page?session_id=custom_1").json["goal"]["asin"] == "A000000001"
    assert adapted.product_item_dict["A000000001"]["MainImage"] == "/minja-assets/adversarial.png"
    write(overlays, {})
    assert adapted.product_item_dict["A000000001"]["MainImage"] == "https://example.test/original.png"
    assert client.get("/minja-assets/adversarial.png").data == b"fixture"
    assert client.get("/minja-assets/../registry.json").status_code == 404
    assert client.get("/minja-disk/health").json["disk_backed"] is True
    assert metadata["scope"] == "full_catalog"
    assert metadata["task_registry"] == str(registry.resolve())
    assert metadata["asset_root"] == str(assets.resolve())
    with pytest.raises(RuntimeError, match="forbids"):
        adapted.load_products()
    adapted.app._minja_disk_catalog.close()


def test_registered_goal_must_be_valid_and_goal_index_is_bound(catalog_root):
    with DiskCatalog.open(catalog_root) as catalog:
        assert full_webshop._goal({"goal_index": 0}, catalog)["asin"] == "A000000001"
        with pytest.raises(ValueError, match="outside"):
            full_webshop._goal(-1, catalog)
        with pytest.raises(ValueError, match="absent"):
            full_webshop._goal(goal("missing"), catalog)
        with pytest.raises(ValueError, match="missing fields"):
            full_webshop._goal({"instruction_text": "display only"}, catalog)


def test_normalization_keeps_empty_image_and_title_products():
    raw = raw_product("A000000001", "")
    raw["images"], raw["small_description"] = [], []
    product = disk_catalog.normalized(raw, {}, None)
    assert product["Title"] == ""
    assert product["MainImage"] == ""
    assert "normal item" in disk_catalog.search_contents(product)


def test_full_build_verifies_and_copies_target_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(disk_catalog, "PROJECT_ROOT", tmp_path)
    items, attrs, human = [tmp_path / name for name in ("items.json", "attrs.json", "human.json")]
    write(items, [raw_product("A000000001", "甲商品")])
    write(attrs, {})
    write(human, {"A000000001": []})
    target = [{"victim": "乙类别", "target": "甲商品", "target_asins": ["A000000001"]}]
    destination = tmp_path / "src" / "output" / "full"
    manifest = build_catalog(items, attrs, human, destination, tmp_path,
                  goal_builder=lambda products, prices: [goal()], search_backend="sqlite_fts5", targets=target)
    assert json.loads((destination / "targets.json").read_text(encoding="utf-8")) == target
    assert manifest["targets_verified"] is True
    assert manifest["targets_sha256"] == fingerprint(target)
    target[0]["target"] = "不匹配商品"
    with pytest.raises(ValueError, match="title does not match"):
        build_catalog(items, attrs, human, tmp_path / "src" / "output" / "bad", tmp_path,
                  goal_builder=lambda products, prices: [goal()], search_backend="sqlite_fts5", targets=target)


def test_default_full_build_dispatches_lucene_without_silent_fts(catalog_root, monkeypatch):
    calls = []
    def fake_index(root, expected, progress):
        calls.append(expected)
        (root / "index").mkdir()
        (root / "index" / "segments_1").write_bytes(b"test dispatch only")
    monkeypatch.setattr(disk_catalog, "_build_lucene_index", fake_index)
    task_root = disk_catalog.PROJECT_ROOT
    destination = task_root / "src" / "output" / "lucene"
    build_catalog(task_root / "items.json", task_root / "attrs.json", task_root / "human.json",
                  destination, task_root, goal_builder=lambda products, prices: [goal()])
    assert calls == [2]
    index = json.loads((destination / "index_manifest.json").read_text(encoding="utf-8"))
    assert index["backend"] == "lucene"
    assert index["path"] == "index"
    assert (destination / "search_documents" / "documents.jsonl").is_file()
    with sqlite3.connect(destination / "catalog.sqlite") as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='products_fts'").fetchone() is None
