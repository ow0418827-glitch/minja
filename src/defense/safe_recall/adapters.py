"""Narrow adapter around the existing WebShop environment.

The defense receives only visible page data. Reward, expected ASIN, attack
labels, catalog files and the original experiment task dictionary stay out.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlsplit

from .contracts import Observation


class ShoppingEnvironment(Protocol):
    def reset(self) -> Observation: ...
    def search(self, query: str) -> Observation: ...
    def open(self, handle: str) -> Observation: ...
    def detail(self, page: str) -> tuple[Observation, Observation]: ...
    def select(self, label: str) -> Observation: ...
    def back(self) -> Observation: ...
    def next_page(self) -> Observation: ...
    def previous_page(self) -> Observation: ...
    def buy(self, expected: Observation) -> tuple[bool, str | None]: ...


class CurrentImageLoader:
    """Read only the image URL observed on the current item page.

    Remote hosts must be configured explicitly. This function never reads a
    historic image path, audit original image, or evaluation directory.
    """

    def __init__(self, allowed_hosts: frozenset[str], *, max_bytes: int = 4_000_000,
                 cache_records: Path | None = None, cache_only: bool = False):
        self.allowed_hosts = allowed_hosts
        self.max_bytes = max_bytes
        self.cache_records = Path(cache_records).resolve() if cache_records else None
        self.cache_only = cache_only

    def _cached(self, asset: dict[str, object], url: str) -> tuple[bytes | None, str | None]:
        if self.cache_records is None:
            return None, None
        asin = asset.get("ASIN")
        if not isinstance(asin, str) or not re.fullmatch(r"[A-Z0-9]{10}", asin):
            return None, None
        try:
            record = json.loads((self.cache_records / f"{asin}.json").read_text(
                encoding="utf-8"))
            if (record.get("asin") != asin or record.get("url") != url
                    or not isinstance(record.get("path"), str)):
                return None, None
            root = self.cache_records.parent.resolve()
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                return None, None
            body = path.read_bytes()
            if (not body or len(body) > self.max_bytes
                    or hashlib.sha256(body).hexdigest() != record.get("sha256")):
                return None, None
            mime = ("image/jpeg" if body.startswith(b"\xff\xd8\xff") else
                    "image/png" if body.startswith(b"\x89PNG\r\n\x1a\n") else
                    "image/webp" if body[:4] == b"RIFF" and body[8:12] == b"WEBP"
                    else None)
            return (body, mime) if mime else (None, None)
        except (OSError, ValueError, TypeError, AttributeError):
            return None, None

    def __call__(self, asset: dict[str, object]) -> tuple[bytes | None, str | None]:
        url = str(asset.get("SourceURL", ""))
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.hostname.lower() not in self.allowed_hosts
                or parsed.username or parsed.password):
            return None, None
        cached = self._cached(asset, url)
        if cached[0] is not None:
            return cached
        if self.cache_only:
            return None, None
        import requests
        try:
            with requests.get(url, timeout=(3, 12), allow_redirects=False,
                              stream=True) as response:
                response.raise_for_status()
                mime = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if mime not in {"image/jpeg", "image/png", "image/webp"}:
                    return None, None
                body = response.raw.read(self.max_bytes + 1)
                if not body or len(body) > self.max_bytes:
                    return None, None
                return body, mime
        except requests.RequestException:
            return None, None


class WebShopAdapter:
    def __init__(self, environment: object, session: str, *,
                 image_loader: Callable[[dict[str, object]],
                                        tuple[bytes | None, str | None]] | None = None):
        self._env = environment
        self._session = session
        self._version = 0
        self._image_loader = image_loader

    def _step(self, action: str) -> Observation:
        text, _reward, _done, info = self._env.step(self._session, action)
        self._version += 1
        state = self._env.sessions[self._session]
        page = str(state.get("page_type", ""))
        handle = str(state.get("asin")) if page in {"item", "item_sub"} else None
        price = info.get("current_item_price") if page == "item" else None
        asset = info.get("current_item_image") if page == "item" else None
        image_bytes, image_mime = (self._image_loader(asset)
                                   if self._image_loader and isinstance(asset, dict)
                                   and asset.get("ASIN") == handle else (None, None))
        option_types = info.get("option_types", {}) if page == "item" else {}
        visible = info.get("asins", ()) if page == "search" else ()
        return Observation(
            page=page, text=str(text), version=self._version,
            page_number=int(state.get("page_num", 1)),
            has_next=bool(info.get("has_next", False)) if page == "search" else False,
            visible_ids=tuple(str(x) for x in visible),
            handle=handle,
            price=float(price) if isinstance(price, (int, float)) else None,
            option_types={str(k): str(v) for k, v in option_types.items()},
            selected_options={str(k): str(v) for k, v in
                              state.get("options", {}).items()} if page == "item" else {},
            image_bytes=image_bytes, image_mime=image_mime,
        )

    def reset(self) -> Observation:
        return self._step("reset")

    def search(self, query: str) -> Observation:
        if not query.strip() or "\n" in query or "]" in query:
            raise ValueError("Unsafe search query")
        return self._step(f"search[{query}]")

    def open(self, handle: str) -> Observation:
        if handle not in self._env.sessions[self._session].get("asins", ()):
            raise ValueError("Product handle was not on the current search page")
        return self._step(f"click[{handle}]")

    def detail(self, page: str) -> tuple[Observation, Observation]:
        if page not in {"Attributes", "Description", "Features"}:
            raise ValueError("Unsupported detail page")
        subpage = self._step(f"click[{page}]")
        item = self._step("click[< Prev]")
        if subpage.handle != item.handle or item.page != "item":
            raise ValueError("Detail observation lost current product binding")
        return subpage, item

    def select(self, label: str) -> Observation:
        available = self._env.sessions[self._session].get("option_types", {})
        if label not in available:
            raise ValueError("Option is not currently visible")
        self._step(f"click[{label}]")
        # The legacy environment shortens the click observation. A fresh
        # current-page observation is required before binding or purchase.
        return self._step("think[reobserve]")

    def back(self) -> Observation:
        return self._step("click[< Prev]")

    def next_page(self) -> Observation:
        state = self._env.sessions[self._session]
        if state.get("page_type") != "search" or not state.get("has_next"):
            raise ValueError("No next search page")
        return self._step("click[Next >]")

    def previous_page(self) -> Observation:
        state = self._env.sessions[self._session]
        if state.get("page_type") != "search" or state.get("page_num", 1) <= 1:
            raise ValueError("No previous search page")
        return self._step("click[< Prev]")

    def buy(self, expected: Observation) -> tuple[bool, str | None]:
        state = self._env.sessions[self._session]
        if (state.get("page_type") != "item"
                or expected.version != self._version
                or expected.handle != state.get("asin")
                or dict(expected.selected_options) != state.get("options", {})):
            raise ValueError("Purchase evidence or item state changed")
        expected = state.get("asin")
        _, _, done, info = self._env.step(self._session, "click[Buy Now]")
        self._version += 1
        # Reward and hidden target never leave this adapter.
        return bool(done), str(info.get("final_asin")) if done and info.get(
            "final_asin") == expected else None
