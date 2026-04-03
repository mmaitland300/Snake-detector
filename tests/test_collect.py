from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
from PIL import Image

import snake_detector.collect as collect_module
from snake_detector.collect import (
    MANIFEST_FIELDS,
    CollectedImageRecord,
    ResolvedTaxon,
    collect_inaturalist_records,
    download_images_from_manifest,
    load_manifest_keys,
    normalize_license_code,
    resolve_inaturalist_taxon,
    resolve_inaturalist_taxon_id,
    write_manifest,
)


@pytest.fixture(autouse=True)
def _zero_inat_observations_page_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "snake_detector.collect._INAT_OBSERVATIONS_PAGE_DELAY_SEC",
        0.0,
    )


def _tiny_png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _record(**overrides) -> CollectedImageRecord:
    base = {
        "label": "snake",
        "provider": "inaturalist",
        "query": "taxon_id",
        "query_value": "85553",
        "resolved_taxon_id": "85553",
        "resolved_taxon_name": "Serpentes",
        "resolved_rank": "suborder",
        "observation_id": "101",
        "image_id": "5001",
        "observation_taxon_id": "85553",
        "observation_taxon_name": "Serpentes",
        "observation_taxon_rank": "suborder",
        "taxon_name": "Serpentes",
        "species_name": "",
        "quality_grade": "research",
        "captive": "false",
        "observed_on": "2024-05-01",
        "observed_time": "2024-05-01T12:00:00-06:00",
        "created_at": "2024-05-02T00:00:00Z",
        "updated_at": "2024-05-03T00:00:00Z",
        "user_id": "42",
        "user_login": "example-user",
        "latitude": "39.7392",
        "longitude": "-104.9903",
        "positional_accuracy": "10",
        "place_guess": "Denver, Colorado, US",
        "photo_license": "CC-BY",
        "observation_license": "CC-BY",
        "attribution": "Example User, iNaturalist",
        "image_url": "https://static.inaturalist.org/photos/5001/large.jpeg",
        "source_page_url": "https://www.inaturalist.org/observations/101",
        "collected_at": "2026-04-02T00:00:00Z",
    }
    base.update(overrides)
    return CollectedImageRecord(**base)


def test_resolve_inaturalist_taxon_id_exact_name_match(monkeypatch) -> None:
    payload = {
        "results": [
            {"id": 26172, "name": "Squamata", "rank": "order"},
            {"id": 85553, "name": "Serpentes", "rank": "suborder"},
        ],
    }
    monkeypatch.setattr("snake_detector.collect._fetch_json", lambda *a, **k: payload)
    assert resolve_inaturalist_taxon_id("Serpentes") == 85553


def test_resolve_inaturalist_taxon_returns_metadata(monkeypatch) -> None:
    payload = {
        "results": [
            {"id": 85553, "name": "Serpentes", "rank": "suborder"},
        ],
    }
    monkeypatch.setattr("snake_detector.collect._fetch_json", lambda *a, **k: payload)

    resolved = resolve_inaturalist_taxon("Serpentes")

    assert resolved.id == 85553
    assert resolved.name == "Serpentes"
    assert resolved.rank == "suborder"


def test_resolve_inaturalist_taxon_id_raises_when_no_exact_match(monkeypatch) -> None:
    payload = {"results": [{"id": 1, "name": "Other"}]}
    monkeypatch.setattr("snake_detector.collect._fetch_json", lambda *a, **k: payload)
    with pytest.raises(ValueError, match="Serpentes"):
        resolve_inaturalist_taxon_id("Serpentes")


def test_collect_inaturalist_records_skips_taxa_lookup_when_taxon_id_given(monkeypatch) -> None:
    payload = {
        "results": [
            {
                "id": 101,
                "observed_on": "2024-05-01",
                "time_observed_at": "2024-05-01T12:34:56-06:00",
                "license_code": "CC-BY",
                "quality_grade": "research",
                "captive": False,
                "created_at": "2024-05-02T00:00:00Z",
                "updated_at": "2024-05-03T00:00:00Z",
                "geojson": {"coordinates": [-104.9903, 39.7392]},
                "positional_accuracy": 15,
                "place_guess": "Denver, Colorado, US",
                "user": {"id": 42, "login": "example-user"},
                "taxon": {"name": "Homo sapiens"},
                "photos": [
                    {
                        "id": 9001,
                        "license_code": "CC-BY",
                        "attribution": "User",
                        "url": "https://static.inaturalist.org/photos/9001/square.jpeg",
                    },
                ],
            },
        ],
    }
    calls: list[tuple[str, dict]] = []

    def fake_fetch_json(url: str, *, params: dict, user_agent: str):
        calls.append((url, params))
        if params.get("page", 1) > 1:
            return {"results": []}
        return payload

    monkeypatch.setattr("snake_detector.collect._fetch_json", fake_fetch_json)

    result = collect_inaturalist_records(label="no_snake", taxon_id=40151, max_images=5)
    records = result.records

    assert len(records) == 1
    assert records[0].query == "taxon_id"
    assert records[0].query_value == "40151"
    assert records[0].resolved_taxon_id == "40151"
    assert records[0].resolved_taxon_name == ""
    assert records[0].user_login == "example-user"
    assert records[0].latitude == "39.7392"
    assert records[0].longitude == "-104.9903"
    assert len(calls) == 2
    assert calls[0][1].get("taxon_id") == 40151


def test_normalize_license_code_handles_empty_values() -> None:
    assert normalize_license_code(None) == ""
    assert normalize_license_code(" cc by-sa ") == "CC-BY-SA"


def test_collect_inaturalist_records_filters_unlicensed_and_duplicate_photos(monkeypatch) -> None:
    payload = {
        "results": [
            {
                "id": 101,
                "observed_on": "2024-05-01",
                "time_observed_at": "2024-05-01T12:00:00-06:00",
                "license_code": "CC-BY",
                "quality_grade": "research",
                "captive": False,
                "created_at": "2024-05-02T00:00:00Z",
                "updated_at": "2024-05-03T00:00:00Z",
                "user": {"id": 12, "login": "snake-fan"},
                "location": "39.7392,-104.9903",
                "positional_accuracy": 5,
                "place_guess": "Denver, Colorado, US",
                "taxon": {"id": 85553, "name": "Serpentes", "rank": "suborder"},
                "photos": [
                    {
                        "id": 5001,
                        "license_code": "CC-BY",
                        "attribution": "Example User, iNaturalist",
                        "url": "https://static.inaturalist.org/photos/5001/square.jpeg",
                    },
                    {
                        "id": 5001,
                        "license_code": "CC-BY",
                        "attribution": "Duplicate",
                        "url": "https://static.inaturalist.org/photos/5001/square.jpeg",
                    },
                    {
                        "id": 5002,
                        "license_code": None,
                        "attribution": "No License",
                        "url": "https://static.inaturalist.org/photos/5002/square.jpeg",
                    },
                ],
            }
        ]
    }

    monkeypatch.setattr("snake_detector.collect._fetch_json", lambda *args, **kwargs: payload)
    monkeypatch.setattr(
        "snake_detector.collect.resolve_inaturalist_taxon",
        lambda *a, **k: ResolvedTaxon(id=85553, name="Serpentes", rank="suborder"),
    )

    result = collect_inaturalist_records(
        label="snake",
        taxon_name="Serpentes",
        max_images=5,
    )
    records = result.records

    assert len(records) == 1
    assert records[0].image_id == "5001"
    assert records[0].query == "taxon_id"
    assert records[0].query_value == "85553"
    assert records[0].resolved_taxon_name == "Serpentes"
    assert records[0].resolved_rank == "suborder"
    assert records[0].observation_taxon_id == "85553"
    assert records[0].observation_taxon_rank == "suborder"
    assert records[0].quality_grade == "research"
    assert records[0].captive == "false"
    assert records[0].user_login == "snake-fan"
    assert records[0].latitude == "39.7392"
    assert records[0].longitude == "-104.9903"
    assert records[0].image_url.endswith("/large.jpeg")


def test_collect_inaturalist_records_keeps_paging_when_page_has_no_licensed_photos(monkeypatch) -> None:
    page1 = {
        "results": [
            {
                "id": 201,
                "observed_on": "2024-05-02",
                "license_code": "CC-BY",
                "taxon": {"id": 85553, "name": "Serpentes", "rank": "suborder"},
                "photos": [
                    {
                        "id": 6001,
                        "license_code": None,
                        "attribution": "No license",
                        "url": "https://static.inaturalist.org/photos/6001/square.jpeg",
                    },
                ],
            },
        ],
    }
    page2 = {
        "results": [
            {
                "id": 202,
                "observed_on": "2024-05-03",
                "license_code": "CC-BY",
                "taxon": {"id": 85553, "name": "Serpentes", "rank": "suborder"},
                "photos": [
                    {
                        "id": 6002,
                        "license_code": "CC-BY",
                        "attribution": "User",
                        "url": "https://static.inaturalist.org/photos/6002/square.jpeg",
                    },
                ],
            },
        ],
    }
    pages = iter([page1, page2])

    def fake_fetch_json(url: str, *, params: dict, user_agent: str):
        _ = (url, params, user_agent)
        return next(pages)

    monkeypatch.setattr("snake_detector.collect._fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        "snake_detector.collect.resolve_inaturalist_taxon",
        lambda *a, **k: ResolvedTaxon(id=85553, name="Serpentes", rank="suborder"),
    )

    result = collect_inaturalist_records(
        label="snake",
        taxon_name="Serpentes",
        max_images=1,
    )
    records = result.records

    assert len(records) == 1
    assert records[0].image_id == "6002"


def test_download_manifest_images_writes_class_folders(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            _record()
        ],
        manifest,
    )
    png = _tiny_png()
    monkeypatch.setattr("snake_detector.collect._fetch_binary", lambda *args, **kwargs: png)

    result = download_images_from_manifest(manifest, tmp_path / "raw")

    assert result == {"downloaded": 1, "skipped_existing": 0, "failed": 0}
    output_path = tmp_path / "raw" / "snake" / "inaturalist_5001.jpeg"
    assert output_path.exists()
    assert output_path.read_bytes() == png


def test_download_manifest_rejects_non_image_bytes(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            _record()
        ],
        manifest,
    )
    monkeypatch.setattr(
        "snake_detector.collect._fetch_binary",
        lambda *args, **kwargs: b"<html><title>403</title></html>",
    )

    result = download_images_from_manifest(manifest, tmp_path / "raw")

    assert result == {"downloaded": 0, "skipped_existing": 0, "failed": 1}
    assert not (tmp_path / "raw" / "snake" / "inaturalist_5001.jpg").exists()


def test_download_manifest_accepts_minimal_png(tmp_path: Path, monkeypatch) -> None:
    png_bytes = _tiny_png()

    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [
            _record(image_url="https://static.inaturalist.org/photos/5001/large.png")
        ],
        manifest,
    )
    monkeypatch.setattr("snake_detector.collect._fetch_binary", lambda *args, **kwargs: png_bytes)

    result = download_images_from_manifest(manifest, tmp_path / "raw")

    assert result == {"downloaded": 1, "skipped_existing": 0, "failed": 0}
    out = tmp_path / "raw" / "snake" / "inaturalist_5001.png"
    assert out.exists()
    assert out.read_bytes() == png_bytes


def test_write_manifest_rejects_append_when_header_schema_changed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("label,provider,image_id\nsnake,inaturalist,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest header mismatch"):
        write_manifest([_record()], manifest, append=True)


def test_write_manifest_uses_current_manifest_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest([_record()], manifest)

    header = manifest.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == MANIFEST_FIELDS


def test_collect_inaturalist_records_handles_missing_optional_metadata(monkeypatch) -> None:
    taxa_payload = {"results": [{"id": 85553, "name": "Serpentes", "rank": "suborder"}]}
    observations_payload = {
        "results": [
            {
                "id": 303,
                "observed_on": "2024-05-04",
                "license_code": "CC-BY",
                "taxon": {"id": 85553, "name": "Serpentes", "rank": "suborder"},
                "photos": [
                    {
                        "id": 7001,
                        "license_code": "CC-BY",
                        "attribution": "User",
                        "url": "https://static.inaturalist.org/photos/7001/square.jpeg",
                    },
                ],
            }
        ]
    }

    def fake_fetch_json(url: str, *, params: dict, user_agent: str):
        _ = (params, user_agent)
        if "taxa" in url:
            return taxa_payload
        return observations_payload

    monkeypatch.setattr("snake_detector.collect._fetch_json", fake_fetch_json)

    result = collect_inaturalist_records(label="snake", taxon_name="Serpentes", max_images=1)
    records = result.records

    assert len(records) == 1
    record = records[0]
    assert record.user_id == ""
    assert record.user_login == ""
    assert record.latitude == ""
    assert record.longitude == ""
    assert record.observed_time == ""
    assert record.species_name == ""


def test_fetch_json_retries_after_http_403(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self.payload

    def fake_urlopen(request, timeout=60):
        _ = (request, timeout)
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                url="https://api.inaturalist.org/v1/observations",
                code=403,
                msg="Forbidden",
                hdrs=None,
                fp=None,
            )
        return FakeResponse(json.dumps({"results": []}).encode("utf-8"))

    monkeypatch.setattr(collect_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(collect_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    payload = collect_module._fetch_json(
        "https://api.inaturalist.org/v1/observations",
        params={"page": 1},
        user_agent="test-agent",
    )

    assert payload == {"results": []}
    assert calls["count"] == 2
    assert sleeps == [2.0]


def test_fetch_json_raises_after_exhausting_retries(monkeypatch) -> None:
    sleeps: list[float] = []

    def fake_urlopen(request, timeout=60):
        _ = (request, timeout)
        raise HTTPError(
            url="https://api.inaturalist.org/v1/observations",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(collect_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(collect_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(HTTPError, match="403"):
        collect_module._fetch_json(
            "https://api.inaturalist.org/v1/observations",
            params={"page": 1},
            user_agent="test-agent",
        )

    assert sleeps == list(collect_module._REQUEST_RETRY_DELAYS_SEC)


def test_fetch_binary_retries_after_url_error(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return b"binary-image"

    def fake_urlopen(request, timeout=60):
        _ = (request, timeout)
        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError("temporary network issue")
        return FakeResponse()

    monkeypatch.setattr(collect_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(collect_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    content = collect_module._fetch_binary("https://example.com/test.jpg", user_agent="test-agent")

    assert content == b"binary-image"
    assert calls["count"] == 2
    assert sleeps == [2.0]


def test_collect_respects_start_page_and_max_pages(monkeypatch) -> None:
    taxa_payload = {"results": [{"id": 85553, "name": "Serpentes", "rank": "suborder"}]}
    obs_payload = {
        "results": [
            {
                "id": 101,
                "observed_on": "2024-05-01",
                "license_code": "CC-BY",
                "quality_grade": "research",
                "captive": False,
                "created_at": "2024-05-02T00:00:00Z",
                "updated_at": "2024-05-03T00:00:00Z",
                "user": {"id": 42, "login": "u"},
                "taxon": {"id": 85553, "name": "Serpentes", "rank": "suborder"},
                "photos": [
                    {
                        "id": 9001,
                        "license_code": "CC-BY",
                        "attribution": "x",
                        "url": "https://static.inaturalist.org/photos/9001/square.jpeg",
                    },
                ],
            },
        ],
    }
    pages_seen: list[int] = []

    def fake_fetch_json(url: str, *, params: dict, user_agent: str):
        _ = user_agent
        if "taxa" in url:
            return taxa_payload
        pages_seen.append(int(params["page"]))
        return obs_payload

    monkeypatch.setattr("snake_detector.collect._fetch_json", fake_fetch_json)

    result = collect_inaturalist_records(
        label="snake",
        taxon_name="Serpentes",
        max_images=100,
        start_page=3,
        max_pages=2,
    )

    assert pages_seen == [3, 4]
    assert result.pages_fetched == 2
    assert result.next_page == 5


def test_collect_max_pages_limits_fetches_despite_high_max_images(monkeypatch) -> None:
    taxa_payload = {"results": [{"id": 85553, "name": "Serpentes", "rank": "suborder"}]}
    obs_payload = {
        "results": [
            {
                "id": 101,
                "observed_on": "2024-05-01",
                "license_code": "CC-BY",
                "quality_grade": "research",
                "captive": False,
                "created_at": "2024-05-02T00:00:00Z",
                "updated_at": "2024-05-03T00:00:00Z",
                "user": {"id": 42, "login": "u"},
                "taxon": {"id": 85553, "name": "Serpentes", "rank": "suborder"},
                "photos": [
                    {
                        "id": 9001,
                        "license_code": "CC-BY",
                        "attribution": "x",
                        "url": "https://static.inaturalist.org/photos/9001/square.jpeg",
                    },
                ],
            },
        ],
    }

    def fake_fetch_json(url: str, *, params: dict, user_agent: str):
        _ = user_agent
        if "taxa" in url:
            return taxa_payload
        return obs_payload

    monkeypatch.setattr("snake_detector.collect._fetch_json", fake_fetch_json)

    result = collect_inaturalist_records(
        label="snake",
        taxon_name="Serpentes",
        max_images=9999,
        max_pages=1,
    )

    assert result.pages_fetched == 1
    assert len(result.records) == 1


def test_write_manifest_append_dedupes_by_label_provider_image_id(tmp_path: Path) -> None:
    manifest = tmp_path / "m.csv"
    write_manifest([_record()], manifest)
    w, s = write_manifest(
        [_record(), _record(image_id="9999")],
        manifest,
        append=True,
        dedupe=True,
    )
    assert s == 1
    assert w == 1
    assert len(load_manifest_keys(manifest)) == 2


def test_flush_every_page_persists_before_later_http_error(tmp_path: Path, monkeypatch) -> None:
    taxa_payload = {"results": [{"id": 85553, "name": "Serpentes", "rank": "suborder"}]}
    obs_payload = {
        "results": [
            {
                "id": 101,
                "observed_on": "2024-05-01",
                "license_code": "CC-BY",
                "quality_grade": "research",
                "captive": False,
                "created_at": "2024-05-02T00:00:00Z",
                "updated_at": "2024-05-03T00:00:00Z",
                "user": {"id": 42, "login": "u"},
                "taxon": {"id": 85553, "name": "Serpentes", "rank": "suborder"},
                "photos": [
                    {
                        "id": 9001,
                        "license_code": "CC-BY",
                        "attribution": "x",
                        "url": "https://static.inaturalist.org/photos/9001/square.jpeg",
                    },
                ],
            },
        ],
    }
    calls = {"obs": 0}

    def fake_fetch_json(url: str, *, params: dict, user_agent: str):
        _ = user_agent
        if "taxa" in url:
            return taxa_payload
        calls["obs"] += 1
        if calls["obs"] == 1:
            return obs_payload
        raise HTTPError(
            "https://api.inaturalist.org/v1/observations",
            403,
            "Forbidden",
            None,
            None,
        )

    monkeypatch.setattr("snake_detector.collect._fetch_json", fake_fetch_json)
    manifest = tmp_path / "checkpoint.csv"
    with pytest.raises(HTTPError):
        collect_inaturalist_records(
            label="snake",
            taxon_name="Serpentes",
            max_images=50,
            max_pages=5,
            manifest_path=manifest,
            flush_every_page=True,
        )
    assert manifest.exists()
    assert len(load_manifest_keys(manifest)) == 1
