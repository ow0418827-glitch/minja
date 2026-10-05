"""Extract WebShop product images and persist them beside a memory JSON file.

This module handles durable image assets. The optional visual_retrieval module
adds image or image/text keys while preserving the original action embeddings.
"""

from __future__ import annotations

import hashlib
import copy
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import unquote, urljoin, urlparse

from PIL import Image, UnidentifiedImageError
import requests


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}
IMAGE_MIMES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}


def validated_image_bytes(record, memory_path=None, *, require_digest=True):
    """Read exactly the audited local image; never fetch or substitute another key."""
    if not isinstance(record, dict) or not record.get("LocalPath"):
        raise ValueError("A local product image is required for multimodal policy input")
    path = Path(record["LocalPath"])
    if not path.is_absolute():
        if memory_path is None:
            raise ValueError("Relative image paths require the containing memory JSON path")
        path = Path(memory_path).resolve().parent / path
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Policy image is missing or exceeds maximum size")
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    expected = record.get("SHA256")
    if require_digest and not expected:
        raise ValueError("Product image SHA256 is required for audited policy input")
    if expected and expected != digest:
        raise ValueError(f"Product image digest mismatch: {path}")
    if not body or len(body) > MAX_IMAGE_BYTES:
        raise ValueError("Policy image is empty or exceeds maximum size")
    try:
        with Image.open(BytesIO(body)) as raster:
            if raster.width * raster.height > MAX_IMAGE_PIXELS:
                raise ValueError("Policy image exceeds maximum pixel count")
            image_format = raster.format
            raster.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError(f"Policy image is not a valid raster: {path}") from exc
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"Unsupported policy image format: {image_format}")
    return path, body, {"SHA256": digest, "Bytes": len(body), "Format": image_format,
                        "MimeType": IMAGE_MIMES[image_format]}


def _purchase_history_digest(entry):
    fields = ('Id', 'Instruction', 'OriginalInstruction', 'Actions', 'Success', 'Reward',
              'EnvironmentReward', 'Completed', 'FinalASIN', 'FinalOptions', 'FinalPrice')
    payload = {field: entry.get(field) for field in fields}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_controlled_image_substitution(entry, memory_path):
    """Audit an explicitly substituted display/key without changing the real purchase."""
    evidence = entry.get('ImageSubstitution')
    allowed = {'controlled_image_substitution', 'attacker_owned_product_image_substitution'}
    if not isinstance(evidence, dict) or evidence.get('protocol') not in allowed:
        raise ValueError('Explicit controlled image substitution evidence is required')
    image_field = evidence.get('imageField')
    if image_field != 'RetrievalImage':
        raise ValueError('Controlled substitution must retain the original PrimaryImage')
    purchased, source = evidence.get('purchasedASIN'), evidence.get('sourceASIN')
    owned = evidence['protocol'] == 'attacker_owned_product_image_substitution'
    if not purchased or not source:
        raise ValueError('Purchased and attack-source identities are required')
    if owned and purchased != source:
        raise ValueError('Attacker-owned perturbation must keep the purchased A product identity')
    if not owned and purchased == source:
        raise ValueError('Purchased and depicted attack-source products must be distinct')
    if entry.get('Completed') is not True or entry.get('FinalASIN') != purchased:
        raise ValueError('Controlled substitution changed the recorded completed purchase')
    if evidence.get('originalHistorySHA') != _purchase_history_digest(entry):
        raise ValueError('Controlled substitution changed the original text or purchase history')
    primary = entry.get('PrimaryImage')
    replacement = entry.get(image_field)
    if not isinstance(primary, dict) or primary.get('ASIN') != purchased:
        raise ValueError('Original primary image must remain bound to the real purchase')
    if (not isinstance(replacement, dict) or replacement.get('ASIN') != purchased
            or replacement.get('DepictedASIN') != source
            or replacement.get('AttackSourceASIN') != source):
        raise ValueError('Substitution must distinguish memory association from depicted product')
    _, _, primary_meta = validated_image_bytes(primary, memory_path)
    _, _, adversarial_meta = validated_image_bytes(replacement, memory_path)
    if (evidence.get('originalSHA') != primary_meta['SHA256']
            or evidence.get('adversarialSHA') != adversarial_meta['SHA256']):
        raise ValueError('Controlled substitution image evidence has changed')
    original = replacement.get('EvidenceOriginalPrimaryImage')
    if not isinstance(original, dict) or original.get('ASIN') != purchased:
        raise ValueError('Substituted image must retain original primary image evidence')
    _, _, original_meta = validated_image_bytes(original, memory_path)
    if original_meta['SHA256'] != primary_meta['SHA256']:
        raise ValueError('Original primary image evidence differs from the real purchase')
    return copy.deepcopy(evidence)


def apply_controlled_image_substitution(entry, adversarial_image_path, memory_path, *,
                                       source_asin, image_field='RetrievalImage',
                                       protocol='controlled_image_substitution'):
    """Replace one audited image while preserving the real purchase and original image.

    The caller applies this to exactly one row.  Pixel-budget/source-image checks
    belong to the attack preparation stage. The attacker-owned protocol requires
    a real A purchase and an adversarial image derived from the same A product.
    """
    if image_field != 'RetrievalImage':
        raise ValueError('Controlled image substitution supports RetrievalImage only')
    if entry.get('ImageSubstitution'):
        raise ValueError('A memory row cannot receive a second controlled substitution')
    primary = entry.get('PrimaryImage')
    purchased = entry.get('FinalASIN')
    if protocol not in {'controlled_image_substitution', 'attacker_owned_product_image_substitution'}:
        raise ValueError('Unsupported image substitution protocol')
    owned = protocol == 'attacker_owned_product_image_substitution'
    if (entry.get('Completed') is not True or not purchased or not source_asin
            or (purchased != source_asin if owned else purchased == source_asin) or not isinstance(primary, dict)
            or primary.get('ASIN') != purchased):
        raise ValueError('Substitution requires a completed purchase consistent with the selected protocol')
    primary_path, _, primary_meta = validated_image_bytes(primary, memory_path)
    adversarial_path, _, adversarial_meta = validated_image_bytes(
        {'LocalPath': str(Path(adversarial_image_path).resolve())}, require_digest=False)
    if primary_meta['SHA256'] == adversarial_meta['SHA256']:
        raise ValueError('Substitution image must differ from the original purchased image')
    poisoned = copy.deepcopy(entry)
    original = copy.deepcopy(primary)
    original['LocalPath'] = str(primary_path)
    poisoned[image_field] = {
        'ASIN': purchased, 'DepictedASIN': source_asin, 'AttackSourceASIN': source_asin,
        'LocalPath': str(adversarial_path), 'Role': 'controlled_image_substitution',
        'EvidenceOriginalPrimaryImage': original, **adversarial_meta,
    }
    poisoned['ImageSubstitution'] = {
        'protocol': protocol, 'imageField': image_field,
        'originalSHA': primary_meta['SHA256'], 'adversarialSHA': adversarial_meta['SHA256'],
        'sourceASIN': source_asin, 'purchasedASIN': purchased,
        'originalHistorySHA': _purchase_history_digest(entry),
    }
    validate_controlled_image_substitution(poisoned, memory_path)
    return poisoned


def memory_image_inputs(memory, selected_ids, memory_path, *, image_field="PrimaryImage"):
    """Resolve each displayed example's purchased image, including normal examples."""
    if image_field not in {"PrimaryImage", "RetrievalImage"}:
        raise ValueError("Policy memory image field must be PrimaryImage or RetrievalImage")
    images = []
    for identifier in selected_ids:
        matches = [entry for entry in memory if entry.get("Id") == identifier]
        if len(matches) != 1:
            raise ValueError(f"Displayed memory must have exactly one entry: {identifier}")
        entry = matches[0]
        for field in ("Id", "Instruction", "Actions", "Success", "Reward"):
            if field not in entry:
                raise ValueError(f"Displayed memory is missing {field}: {identifier}")
        asin = entry.get("FinalASIN")
        record = entry.get(image_field)
        if not asin or not isinstance(record, dict) or record.get("ASIN") != asin:
            raise ValueError(f"Policy image is not bound to the recorded purchase: {identifier}")
        substitution = (validate_controlled_image_substitution(entry, memory_path)
                        if entry.get('ImageSubstitution') is not None else None)
        if (record.get('DepictedASIN', asin) != asin
                or record.get('AttackSourceASIN', asin) != asin):
            if substitution is None or substitution['imageField'] != image_field:
                raise ValueError('Cross-product images require explicit substitution evidence')
        path, _, metadata = validated_image_bytes(record, memory_path)
        asset = {"Id": identifier, "ASIN": asin, "LocalPath": str(path),
                 "ImageField": image_field, "Role": "recalled_memory", **metadata}
        if substitution and substitution['imageField'] == image_field:
            asset.update(DepictedASIN=record['DepictedASIN'], AttackSourceASIN=record['AttackSourceASIN'],
                         ImageSubstitution=substitution)
        images.append(asset)
    return images


def persist_local_image(image, memory_path, image_dir=None, *, source_memory_path=None):
    """Copy the observed bytes into the bank instead of redownloading a mutable URL."""
    _, body, metadata = validated_image_bytes(image, source_memory_path or memory_path)
    memory_path = Path(memory_path).resolve()
    image_dir = Path(image_dir).resolve() if image_dir else memory_path.parent / "memory_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    destination = image_dir / (metadata["SHA256"] + IMAGE_FORMATS[metadata["Format"]])
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != metadata["SHA256"]:
            raise ValueError("Existing image asset has changed")
    else:
        with tempfile.NamedTemporaryFile(dir=image_dir, prefix=".image-", delete=False) as tmp:
            temporary_path = Path(tmp.name)
            tmp.write(body)
        try:
            os.replace(temporary_path, destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    saved = dict(image, **metadata)
    saved["LocalPath"] = Path(os.path.relpath(destination, memory_path.parent)).as_posix()
    return saved


def _assert_original_product_image(record, asin):
    role = str(record.get('Role', record.get('role', ''))).lower()
    if (record.get('DepictedASIN', asin) != asin or record.get('AttackSourceASIN', asin) != asin
            or record.get('ImageSubstitution') or record.get('EvidenceOriginalPrimaryImage')
            or record.get('is_adversarial') or any(word in role for word in ('adversarial', 'attack', 'substitution'))):
        raise ValueError('Attack or substituted images cannot enter the original product cache')


def _cached_original_image(cache_root, record_path, image):
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding='utf-8'))
    asin = image['ASIN']
    if not isinstance(record, dict) or record.get('asin') != asin:
        raise ValueError('Original product cache ASIN does not match the observed product')
    _assert_original_product_image(record, asin)
    original_url = record.get('url')
    if not isinstance(original_url, str) or urlparse(original_url).scheme not in ('http', 'https'):
        raise ValueError('Original product cache must record its original URL')
    raw_path = record.get('path')
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError('Original product cache has no asset path')
    assets = (cache_root / 'assets').resolve()
    path = Path(raw_path)
    path = (path if path.is_absolute() else cache_root / path).resolve()
    if not path.is_relative_to(assets):
        raise ValueError('Original product cache asset escapes its assets directory')
    observed_url = image.get('SourceURL')
    if observed_url != original_url:
        parsed = urlparse(observed_url or '')
        relative = path.relative_to(assets).as_posix()
        local_paths = {'/minja-assets/' + relative, '/local-pair/assets/' + relative}
        if not (parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}
                and unquote(parsed.path) in local_paths and not parsed.query and not parsed.fragment):
            # Another photo/version of the same product must be fetched separately.
            return None
    asset = {'ASIN': asin, 'LocalPath': str(path), 'SHA256': record.get('sha256')}
    _, _, metadata = validated_image_bytes(asset)
    return dict(image, LocalPath=str(path), OriginalURL=original_url,
                OriginalImageCacheRecord=str(record_path), **metadata)


def snapshot_product_image(image, memory_path, *, image_cache_root,
                           timeout=(10, 30), max_attempts=2):
    """Use an audited prepared/original cache, otherwise fetch and cache exact page bytes."""
    asin = image.get('ASIN')
    if not isinstance(asin, str) or not asin or any(c not in
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in asin):
        raise ValueError('Original product cache requires a safe nonempty ASIN')
    _assert_original_product_image(image, asin)
    source_url = image.get('SourceURL')
    parsed = urlparse(source_url or '')
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('Original product cache requires the observed product URL')
    cache_root = Path(image_cache_root).resolve()
    prepared_path = cache_root / 'image_records' / (asin + '.json')
    prepared = _cached_original_image(cache_root, prepared_path, image)
    if prepared is not None:
        return persist_local_image(dict(prepared, ImageCacheSource='prepared'), memory_path)
    persistent_root = cache_root / 'original_image_cache'
    key = hashlib.sha256((asin + '\0' + source_url).encode('utf-8')).hexdigest()
    index = persistent_root / 'image_records' / (key + '.json')
    cached = _cached_original_image(persistent_root, index, image)
    if cached is not None:
        return persist_local_image(dict(cached, ImageCacheSource='persistent'), memory_path)
    # Unknown local overrides could be adversarial assets; only recorded original
    # localhost assets above may be treated as original-product cache entries.
    if parsed.hostname in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('Unknown local product images require an audited original cache record')
    downloaded = save_memory_image(dict(image, OriginalURL=source_url), persistent_root / 'cache.json',
                                   image_dir=persistent_root / 'assets', timeout=timeout,
                                   max_attempts=max_attempts)
    validated_image_bytes(downloaded, persistent_root / 'cache.json')
    index.parent.mkdir(parents=True, exist_ok=True)
    record = {'asin': asin, 'url': source_url, 'path': downloaded['LocalPath'],
              'sha256': downloaded['SHA256'], 'bytes': downloaded['Bytes'], 'role': 'original_product'}
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=index.parent,
                                     prefix='.image-record-', delete=False) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(record, temporary, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary_path, index)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    downloaded['LocalPath'] = str((persistent_root / downloaded['LocalPath']).resolve())
    downloaded.update(ImageCacheSource='download', OriginalImageCacheRecord=str(index))
    return persist_local_image(downloaded, memory_path)


def _download_image_body(source_url, *, timeout, max_attempts):
    if max_attempts < 1:
        raise ValueError('Image download retry limit must be positive')
    for attempt in range(max_attempts):
        try:
            with requests.get(source_url,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; WebShopMemory/1.0)'},
                    timeout=timeout, stream=True) as response:
                response.raise_for_status()
                declared_size = response.headers.get('Content-Length')
                if declared_size and int(declared_size) > MAX_IMAGE_BYTES:
                    raise ValueError('Image exceeds maximum download size')
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES:
                        raise ValueError('Image exceeds maximum download size')
                    chunks.append(chunk)
                return b''.join(chunks)
        except (requests.RequestException, OSError):
            if attempt + 1 == max_attempts:
                raise


def extract_page_images(html_obj, page_type, asin="", page_url=""):
    """Return product-linked image metadata from a search or item page."""
    images = []
    if page_type == "search":
        for card in html_obj.select(".list-group-item"):
            product = card.select_one("a.product-link")
            image = card.select_one("img.result-img[src]")
            if product is None or image is None:
                continue
            product_asin = product.get_text(strip=True)
            raw_src = (image.get("src") or "").strip()
            if not product_asin or not raw_src:
                continue
            source_url = urljoin(page_url, raw_src)
            title = card.select_one(".product-title")
            images.append({
                "ASIN": product_asin,
                "Title": title.get_text(" ", strip=True) if title else "",
                "SourceURL": source_url,
                "PageType": "search",
            })
    elif page_type == "item":
        image = html_obj.select_one("img#product-image[src]")
        if image is not None and asin:
            raw_src = (image.get("src") or "").strip()
            if raw_src:
                source_url = urljoin(page_url, raw_src)
                title = html_obj.select_one("h2")
                images.append({
                    "ASIN": asin,
                    "Title": title.get_text(" ", strip=True) if title else "",
                    "SourceURL": source_url,
                    "PageType": "item",
                })
    return images


def save_memory_image(image, memory_path, image_dir=None, *, timeout=(5, 20), max_attempts=1):
    """Download one raster image and return JSON-safe, portable metadata.

    ``LocalPath`` is relative to the directory containing ``memory_path``.
    The SHA-256 digest names the file, so identical images share a cached copy.
    Download and validation failures propagate to the caller, which can keep a
    text-only memory entry and record the source URL for a later retry.
    """
    source_url = image.get("SourceURL", "")
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Unsupported image URL: {source_url!r}")

    memory_path = Path(memory_path).resolve()
    image_dir = Path(image_dir).resolve() if image_dir else memory_path.parent / "memory_images"

    body = _download_image_body(source_url, timeout=timeout, max_attempts=max_attempts)
    size = len(body)
    if not body:
        raise ValueError("Downloaded image is empty")
    try:
        from io import BytesIO

        with Image.open(BytesIO(body)) as raster:
            if raster.width * raster.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds maximum pixel count")
            image_format = raster.format
            raster.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError("Downloaded content is not a valid raster image") from exc
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {image_format}")

    digest = hashlib.sha256(body).hexdigest()
    image_dir.mkdir(parents=True, exist_ok=True)
    destination = image_dir / f"{digest}{IMAGE_FORMATS[image_format]}"
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError('Existing downloaded image asset has changed')
    if not destination.exists():
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=image_dir, prefix=".image-", delete=False) as tmp:
                temporary_path = Path(tmp.name)
                tmp.write(body)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()

    saved = dict(image)
    saved["LocalPath"] = Path(os.path.relpath(destination, memory_path.parent)).as_posix()
    saved["SHA256"] = digest
    saved["Format"] = image_format
    saved["Bytes"] = size
    return saved
