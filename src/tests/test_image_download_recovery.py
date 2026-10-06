"""Offline contracts for bounded image transport recovery and honest collection."""

import copy
import hashlib

import pytest
import requests

from src.environment.local_webshop import read
from src.experiments import nine_group_experiment as driver
from src.memory import multimodal_memory as images
from src.tests.test_collection_recovery import collection_fixture, memory
from src.tests.test_original_image_cache import Response, observed, png_bytes


@pytest.fixture(autouse=True)
def offline_transport(monkeypatch):
    def forbidden_request(*args, **kwargs):
        pytest.fail("Image recovery tests must never make a network request")

    monkeypatch.setattr(images.requests, "get", forbidden_request)
    delays = []
    monkeypatch.setattr(images.time, "sleep", delays.append)
    return delays


def eof_error():
    return requests.exceptions.SSLError(
        "SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] "
        "EOF occurred in violation of protocol')"
    )


def status_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"HTTP status {status}", response=response)


def assert_certificate_verification_enabled(calls):
    assert calls
    assert all(options.get("verify", True) is True for _, options in calls)


def test_eof_then_success_keeps_exact_bytes_and_reuses_audited_cache(
    tmp_path, monkeypatch, offline_transport
):
    body, calls = png_bytes(), []
    response = Response(body)

    def download(url, **options):
        calls.append((url, options))
        if len(calls) == 1:
            raise eof_error()
        return response

    monkeypatch.setattr(images.requests, "get", download)
    cache = tmp_path / "cache"
    bank = tmp_path / "first" / "memory.json"
    source = observed()
    saved = images.snapshot_product_image(
        source, bank, image_cache_root=cache, timeout=(0.01, 0.02), max_attempts=2
    )

    assert len(calls) == 2 and offline_transport == [0.5]
    assert all(url == source["SourceURL"] for url, _ in calls)
    assert all(options["timeout"] == (0.01, 0.02) for _, options in calls)
    assert_certificate_verification_enabled(calls)
    assert response.closed is True
    assert saved["ImageCacheSource"] == "download"
    assert images.validated_image_bytes(saved, bank)[1] == body
    assert saved["SHA256"] == hashlib.sha256(body).hexdigest()
    record = read(saved["OriginalImageCacheRecord"])
    assert record["asin"] == source["ASIN"] and record["url"] == source["SourceURL"]
    assert record["role"] == "original_product" and record["sha256"] == saved["SHA256"]

    def no_additional_download(*args, **kwargs):
        pytest.fail("The verified persistent cache must serve the second bank")

    monkeypatch.setattr(images.requests, "get", no_additional_download)
    second_bank = tmp_path / "second" / "memory.json"
    cached = images.snapshot_product_image(source, second_bank, image_cache_root=cache)
    assert cached["ImageCacheSource"] == "persistent"
    assert cached["SHA256"] == saved["SHA256"]
    assert images.validated_image_bytes(cached, second_bank)[1] == body


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.SSLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"),
        requests.exceptions.SSLError("hostname mismatch"),
        requests.exceptions.SSLError("unsupported SSL protocol"),
        status_error(404),
        OSError("local or unclassified I/O error"),
    ],
    ids=["certificate", "hostname", "unknown_ssl", "http_404", "io_error"],
)
def test_permanent_failures_stop_immediately_without_cache_or_disabled_verification(
    tmp_path, monkeypatch, offline_transport, error
):
    calls = []

    def download(url, **options):
        calls.append((url, options))
        raise error

    monkeypatch.setattr(images.requests, "get", download)
    with pytest.raises(type(error)) as caught:
        images.snapshot_product_image(
            observed(), tmp_path / "bank" / "memory.json", image_cache_root=tmp_path / "cache",
            max_attempts=5,
        )
    assert caught.value is error
    assert images.transient_image_error_reason(error) is None
    assert len(calls) == 1 and offline_transport == []
    assert_certificate_verification_enabled(calls)
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.png"))


@pytest.mark.parametrize("attempts, delays", [(1, []), (3, [0.5, 1.0]), (5, [0.5, 1.0, 2.0, 2.0])])
def test_exhausted_eof_retries_preserve_cause_and_create_no_cache(
    tmp_path, monkeypatch, offline_transport, attempts, delays
):
    calls, failures = [], []

    def download(url, **options):
        calls.append((url, options))
        error = eof_error()
        failures.append(error)
        raise error

    monkeypatch.setattr(images.requests, "get", download)
    with pytest.raises(images.RecoverableImageDownloadError) as caught:
        images.snapshot_product_image(
            observed(), tmp_path / "bank" / "memory.json", image_cache_root=tmp_path / "cache",
            timeout=(0.01, 0.02), max_attempts=attempts,
        )
    error = caught.value
    assert isinstance(error, requests.ConnectionError) and error.recoverable is True
    assert error.download_reason == "image_secure_connection_interrupted"
    assert error.download_attempts == attempts and error.__cause__ is failures[-1]
    assert len(calls) == attempts and offline_transport == delays
    assert_certificate_verification_enabled(calls)
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.png"))


@pytest.mark.parametrize(
    "error, reason",
    [
        (requests.Timeout("request timed out"), "image_request_timeout"),
        (requests.ConnectionError("connection reset"), "image_connection_interrupted"),
        (requests.exceptions.ChunkedEncodingError("stream interrupted"), "image_response_stream_interrupted"),
        *[(status_error(status), f"image_http_status_{status}") for status in (408, 429, 500, 502, 503, 504)],
    ],
)
def test_other_transient_transport_failures_are_bounded_and_typed(
    tmp_path, monkeypatch, offline_transport, error, reason
):
    calls = []

    def download(url, **options):
        calls.append((url, options))
        raise error

    monkeypatch.setattr(images.requests, "get", download)
    with pytest.raises(images.RecoverableImageDownloadError) as caught:
        images.save_memory_image(observed(), tmp_path / "memory.json", max_attempts=2)
    assert caught.value.download_reason == reason
    assert caught.value.download_attempts == 2 and caught.value.__cause__ is error
    assert len(calls) == 2 and offline_transport == [0.5]
    assert not list(tmp_path.rglob("*.json"))


def install_image_collection(tmp_path, monkeypatch, rt, stage, download):
    """Use real snapshot validation while keeping shopping and policy calls offline."""
    calls = []
    monkeypatch.setattr(images.requests, "get", download)

    def snapshot(row, bank_path):
        row["PrimaryImage"] = images.snapshot_product_image(
            observed(asin=row["FinalASIN"]), bank_path, image_cache_root=tmp_path / "cache",
            timeout=(0.01, 0.02), max_attempts=2,
        )
        row["Images"] = [copy.deepcopy(row["PrimaryImage"])]

    def execute(args, root, group, task, bank, bank_path, runtime):
        calls.append(task["session"])
        row = memory(task["session"])
        if stage == "episode":
            snapshot(row, bank_path)
        return 1, row, True

    def persist(row, bank_path, *, strict):
        assert strict is True
        if stage == "image_persistence":
            snapshot(row, bank_path)
        stored = images.persist_local_image(row["PrimaryImage"], bank_path)
        images.validated_image_bytes(stored, bank_path)
        row["PrimaryImage"] = stored
        row["Images"] = [copy.deepcopy(stored)]

    monkeypatch.setattr(driver, "execute_episode", execute)
    rt["agent"].persist_memory_images = persist
    return calls


@pytest.mark.parametrize("stage", ["episode", "image_persistence"])
def test_transient_download_failure_preserves_thirteen_histories_and_completes_within_budget(
    tmp_path, monkeypatch, offline_transport, stage
):
    args, group, folder, original_state, rt = collection_fixture(tmp_path, monkeypatch)
    previous = copy.deepcopy(original_state["normal"])
    downloads, body = [], png_bytes()

    def download(url, **options):
        downloads.append((url, options))
        if len(downloads) <= 2:
            raise eof_error()
        return Response(body)

    episodes = install_image_collection(tmp_path, monkeypatch, rt, stage, download)
    driver.collect(args)
    state = read(folder / "history_state.json")
    assert state["normal"][:13] == previous
    assert len(state["normal"]) == 14 and state["designated"] is not None
    assert len(episodes) == 3 and len(set(episodes)) == 3
    assert len(downloads) == 3 and offline_transport == [0.5]
    assert_certificate_verification_enabled(downloads)
    assert [attempt["technical_error"] for attempt in state["attempts"]] == [True, False, False]
    failed = state["attempts"][0]
    assert failed["error"] == "RecoverableImageDownloadError"
    assert failed["error_stage"] == stage and failed["recoverable"] is True
    assert failed["technical_error_reason"] == "image_secure_connection_interrupted"
    assert failed["image_download_attempts"] == 2
    assert failed["accepted"] is False and failed["completed"] is False and failed["reward"] is None
    accepted = read(folder / "history_memory.json")
    assert len(accepted) == 15 and all(row["Id"] != failed["session"] for row in accepted)
    for row in (state["normal"][-1], state["designated"]):
        assert row["RetrievalImage"] == row["PrimaryImage"]
        assert images.validated_image_bytes(row["PrimaryImage"], folder / "history_memory.json")[1] == body
    failures = list((folder / "collection_errors").glob("*.json"))
    assert len(failures) == 1
    saved = read(failures[0])
    if stage == "image_persistence":
        assert saved["trajectory"]["Id"] == failed["session"]
        assert saved["trajectory"]["Completed"] is True
    else:
        assert saved["trajectory"] is None


@pytest.mark.parametrize("stage", ["episode", "image_persistence"])
@pytest.mark.parametrize("failure_kind", ["invalid_image", "io_error"])
def test_invalid_image_and_io_failures_stop_collection_without_accepting_failed_row(
    tmp_path, monkeypatch, offline_transport, stage, failure_kind
):
    args, group, folder, original_state, rt = collection_fixture(tmp_path, monkeypatch)
    previous = copy.deepcopy(original_state["normal"])
    downloads = []

    def download(url, **options):
        downloads.append((url, options))
        if failure_kind == "io_error":
            raise OSError("unclassified I/O fault")
        return Response(b"invalid image content")

    install_image_collection(tmp_path, monkeypatch, rt, stage, download)
    expected = OSError if failure_kind == "io_error" else ValueError
    with pytest.raises(expected):
        driver.collect(args)
    state = read(folder / "history_state.json")
    assert state["normal"] == previous and state["designated"] is None
    assert len(state["attempts"]) == 1
    failed = state["attempts"][0]
    assert failed["error_stage"] == stage and failed["recoverable"] is False
    assert failed["accepted"] is False and failed["reward"] is None
    assert read(folder / "history_memory.json") == previous
    assert len(downloads) == 1 and offline_transport == []
    assert not (folder / "history_binding.json").exists()
    assert not list((tmp_path / "cache").rglob("*.json"))
    assert len(list((folder / "collection_errors").glob("*.json"))) == 1
