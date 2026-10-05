"""Focused tests for WebShop image metadata and durable memory assets."""

import hashlib
from io import BytesIO
import importlib
import json
import os
from pathlib import Path
import sys

import pytest
from bs4 import BeautifulSoup
from PIL import Image

from src.memory import multimodal_memory


@pytest.fixture(scope="module")
def agent_module():
    """Load the agent through its functional package."""
    return importlib.import_module("src.agent.minja")


def test_extract_search_images_keeps_product_associations():
    # Matches the product card structure in WebShop's results_page.html.
    html = BeautifulSoup(
        """
        <html><body>
          <img src="/static/logo.png" alt="site logo">
          <div class="list-group-item">
            <img src="/images/one.jpg" class="result-img">
            <h4 class="product-asin"><a class="product-link">ASIN-ONE</a></h4>
            <h4 class="product-title">Blue cotton shirt</h4>
          </div>
          <div class="list-group-item">
            <img src="https://images.example/two.jpg" class="result-img">
            <h4 class="product-asin"><a class="product-link">ASIN-TWO</a></h4>
            <h4 class="product-title">Red wool shirt</h4>
          </div>
        </body></html>
        """,
        "html.parser",
    )

    images = multimodal_memory.extract_page_images(
        html,
        "search",
        page_url="http://localhost:3000/search_results/session/shirt/1",
    )

    assert len(images) == 2
    assert [image["ASIN"] for image in images] == ["ASIN-ONE", "ASIN-TWO"]
    assert [image["Title"] for image in images] == [
        "Blue cotton shirt",
        "Red wool shirt",
    ]
    assert [image["SourceURL"] for image in images] == [
        "http://localhost:3000/images/one.jpg",
        "https://images.example/two.jpg",
    ]
    assert all(image["PageType"] == "search" for image in images)


def test_extract_item_image_uses_selected_asin():
    # Matches the primary image and title in WebShop's item_page.html.
    html = BeautifulSoup(
        """
        <html><body>
          <img src="/static/logo.png" alt="site logo">
          <img id="product-image" src="//images.example/main.png" class="item-page-img">
          <div class="col-md-6"><h2>Selected product</h2></div>
        </body></html>
        """,
        "html.parser",
    )

    images = multimodal_memory.extract_page_images(
        html,
        "item",
        asin="ASIN-SELECTED",
        page_url="https://shop.example/item_page/session/ASIN-SELECTED",
    )

    assert len(images) == 1
    assert images[0]["ASIN"] == "ASIN-SELECTED"
    assert images[0]["Title"] == "Selected product"
    assert images[0]["SourceURL"] == "https://images.example/main.png"
    assert images[0]["PageType"] == "item"


def test_save_memory_image_persists_bytes_and_relative_path(tmp_path, monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buffer, format="PNG")
    body = buffer.getvalue()
    calls = []

    class FakeResponse:
        content = body
        headers = {"Content-Type": "image/png"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            for start in range(0, len(body), chunk_size):
                yield body[start : start + chunk_size]

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(multimodal_memory.requests, "get", fake_get)
    memory_path = tmp_path / "output" / "memory.json"
    image_dir = tmp_path / "output" / "memory_images"
    image = {
        "ASIN": "ASIN-SELECTED",
        "Title": "Selected product",
        "SourceURL": "https://images.example/main.png",
        "PageType": "item",
    }

    saved = multimodal_memory.save_memory_image(
        image, memory_path, image_dir=image_dir
    )

    assert calls and calls[0][0] == image["SourceURL"]
    assert saved["ASIN"] == image["ASIN"]
    assert saved["SourceURL"] == image["SourceURL"]
    assert saved["SHA256"] == hashlib.sha256(body).hexdigest()
    assert not Path(saved["LocalPath"]).is_absolute()
    local_path = (memory_path.parent / saved["LocalPath"]).resolve()
    assert local_path.parent == image_dir.resolve()
    assert local_path.read_bytes() == body
    assert json.loads(json.dumps(saved))["LocalPath"] == saved["LocalPath"]

    saved_again = multimodal_memory.save_memory_image(
        image, memory_path, image_dir=image_dir
    )
    assert saved_again["LocalPath"] == saved["LocalPath"]
    assert saved_again["SHA256"] == saved["SHA256"]
    assert len(list(image_dir.iterdir())) == 1


def test_save_memory_image_propagates_download_failure(tmp_path, monkeypatch):
    def fake_get(url, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(multimodal_memory.requests, "get", fake_get)
    image = {
        "ASIN": "ASIN-SELECTED",
        "Title": "Selected product",
        "SourceURL": "https://images.example/missing.png",
        "PageType": "item",
    }

    with pytest.raises(OSError, match="network unavailable"):
        multimodal_memory.save_memory_image(
            image, tmp_path / "memory.json", image_dir=tmp_path / "images"
        )

    assert not list(tmp_path.rglob("*.png"))


def test_env_buy_preserves_selected_product_image(agent_module, monkeypatch):
    selected_image = {
        "ASIN": "ASIN-SELECTED",
        "Title": "Selected product",
        "SourceURL": "https://images.example/selected.png",
        "PageType": "item",
    }
    other_image = {
        "ASIN": "ASIN-OTHER",
        "Title": "Other product",
        "SourceURL": "https://images.example/other.png",
        "PageType": "search",
    }

    def fake_webshop_text(*, page_type, **kwargs):
        info = {"instruction": "Buy the selected product"}
        if page_type == "search":
            info.update(asins=["ASIN-OTHER", "ASIN-SELECTED"], images=[other_image, selected_image])
        elif page_type == "item":
            info.update(current_item_image=dict(selected_image), images=[dict(selected_image)])
        elif page_type == "end":
            info["reward"] = 1.0
        return f"Observation for {page_type}", info

    monkeypatch.setattr(agent_module, "webshop_text", fake_webshop_text)
    env = agent_module.webshopEnv()
    env.step("session", "reset")
    env.step("session", "search[selected product]")
    env.step("session", "click[ASIN-SELECTED]")
    _, reward, done, info = env.step("session", "click[Buy Now]")

    assert done is True
    assert reward == 1.0
    assert info["final_asin"] == "ASIN-SELECTED"
    assert info["final_image"]["ASIN"] == "ASIN-SELECTED"
    assert info["final_image"]["SourceURL"] == selected_image["SourceURL"]
    assert info["final_image"]["PageType"] == "item"
    assert info["final_image"]["Step"] == 3


def test_completed_trajectory_returns_image_memory_without_extra_model_call(
    agent_module, monkeypatch
):
    selected_image = {
        "ASIN": "ASIN-SELECTED",
        "Title": "Selected product",
        "SourceURL": "https://images.example/selected.png",
        "PageType": "item",
    }

    def fake_webshop_text(*, page_type, **kwargs):
        info = {"instruction": "Buy a toothbrush"}
        if page_type == "init":
            return "Instruction: Buy a toothbrush", info
        if page_type == "search":
            info["asins"] = ["ASIN-SELECTED"]
            return "Search result: selected product", info
        if page_type == "item":
            info["current_item_image"] = dict(selected_image)
            return "Selected product details", info
        if page_type == "end":
            info["reward"] = 1.0
            return "Your score (min 0.0, max 1.0): 1.0", info
        raise AssertionError(f"Unexpected page type: {page_type}")

    model_actions = iter(
        ["search[selected product]", "click[ASIN-SELECTED]", "click[Buy Now]"]
    )
    model_calls = []

    def fake_llm(*args, **kwargs):
        model_calls.append((args, kwargs))
        return next(model_actions)

    monkeypatch.setattr(agent_module, "webshop_text", fake_webshop_text)
    monkeypatch.setattr(agent_module, "llm", fake_llm)

    reward, data, hit_target = agent_module.webshop_run_rap(
        idx="session",
        victim="toothbrush",
        target="selected product",
        prompt="Test prompt",
        env=agent_module.webshopEnv(),
        client=None,
        model_name="fake-model",
        provider="openai",
        model_embedding=None,
        memory=[],
        embeddings={},
        num_steps=4,
        to_print=False,
    )

    assert len(model_calls) == 3
    assert hit_target is True
    assert reward == 1.0
    assert data["Id"] == "session"
    assert data["Instruction"] == "Buy a toothbrush"
    assert data["Success"] is True
    assert data["Reward"] == 1.0
    assert "Action: search[selected product]" in data["Actions"]
    assert "Action: click[Buy Now]" in data["Actions"]
    assert data["FinalASIN"] == "ASIN-SELECTED"
    assert data["PrimaryImage"]["SourceURL"] == selected_image["SourceURL"]
    assert data["Images"] == [data["PrimaryImage"]]


def test_image_download_failure_keeps_text_memory(agent_module, monkeypatch, tmp_path):
    def fail_to_save(*args, **kwargs):
        raise OSError("image host unavailable")

    monkeypatch.setattr(agent_module, "save_memory_image", fail_to_save)
    memory = {
        "Id": "session",
        "Instruction": "Buy the selected product",
        "Actions": ["Action: search[selected product]"],
        "Success": True,
        "Reward": 1.0,
        "FinalASIN": "ASIN-SELECTED",
        "PrimaryImage": {
            "ASIN": "ASIN-SELECTED",
            "Title": "Selected product",
            "SourceURL": "https://images.example/selected.png",
            "PageType": "item",
        },
    }
    original_text = {key: memory[key] for key in ("Id", "Instruction", "Actions", "Success", "Reward")}

    agent_module.persist_memory_images(memory, tmp_path / "memory.json")

    assert {key: memory[key] for key in original_text} == original_text
    assert memory["PrimaryImage"]["LocalPath"] is None
    assert "image host unavailable" in memory["PrimaryImage"]["StorageError"]
    assert memory["Images"] == [memory["PrimaryImage"]]
    assert json.loads(json.dumps(memory))["Instruction"] == original_text["Instruction"]
