from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from PIL import Image

from .config import ensure_parent
from .data import RAW_IMAGE_EXTENSIONS

INAT_OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"
INAT_TAXA_URL = "https://api.inaturalist.org/v1/taxa"
# Cap API pages per collection run (avoids infinite loops if the API misbehaves).
_MAX_INAT_PAGES = 5000
DEFAULT_USER_AGENT = "snake-detector/0.1 (+https://github.com/mmaitland300/Snake-detector)"
DEFAULT_PUBLIC_LICENSES = ("CC0", "CC-BY", "CC-BY-SA")
# iNaturalist's edge sometimes returns 403 for bare urllib-style requests; these headers match typical API clients.
_INAT_REFERER = "https://www.inaturalist.org/"
_REQUEST_TIMEOUT_SEC = 60
_RETRYABLE_HTTP_STATUS_CODES = frozenset({403, 429, 500, 502, 503, 504})
# One initial attempt plus len(delays) retries; longer gaps help edge 403/WAF cool-downs.
_REQUEST_RETRY_DELAYS_SEC = (2.0, 5.0, 15.0, 45.0, 90.0)
# Pause between observation pages to avoid hammering the API on long runs.
_INAT_OBSERVATIONS_PAGE_DELAY_SEC = 2.0
MANIFEST_FIELDS = [
    "label",
    "provider",
    "query",
    "query_value",
    "resolved_taxon_id",
    "resolved_taxon_name",
    "resolved_rank",
    "observation_id",
    "image_id",
    "observation_taxon_id",
    "observation_taxon_name",
    "observation_taxon_rank",
    "taxon_name",
    "species_name",
    "quality_grade",
    "captive",
    "observed_on",
    "observed_time",
    "created_at",
    "updated_at",
    "user_id",
    "user_login",
    "latitude",
    "longitude",
    "positional_accuracy",
    "place_guess",
    "photo_license",
    "observation_license",
    "attribution",
    "image_url",
    "source_page_url",
    "collected_at",
]


@dataclass(slots=True)
class CollectedImageRecord:
    label: str
    provider: str
    query: str
    query_value: str
    resolved_taxon_id: str
    resolved_taxon_name: str
    resolved_rank: str
    observation_id: str
    image_id: str
    observation_taxon_id: str
    observation_taxon_name: str
    observation_taxon_rank: str
    taxon_name: str
    species_name: str
    quality_grade: str
    captive: str
    observed_on: str
    observed_time: str
    created_at: str
    updated_at: str
    user_id: str
    user_login: str
    latitude: str
    longitude: str
    positional_accuracy: str
    place_guess: str
    photo_license: str
    observation_license: str
    attribution: str
    image_url: str
    source_page_url: str
    collected_at: str


@dataclass(slots=True)
class CollectInaturalistResult:
    """Outcome of ``collect_inaturalist_records`` (all rows collected this run, plus resume hints)."""

    records: list[CollectedImageRecord]
    pages_fetched: int
    start_page: int
    next_page: int
    manifest_rows_written: int = 0
    manifest_rows_skipped_duplicate: int = 0


@dataclass(slots=True)
class ResolvedTaxon:
    id: int
    name: str
    rank: str


def _inaturalist_request_headers(user_agent: str, *, accept: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": accept,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": _INAT_REFERER,
    }


def normalize_license_code(raw: str | None) -> str:
    if not raw:
        return ""
    code = str(raw).strip().upper().replace("_", "-")
    code = re.sub(r"\s+", "-", code)
    return code


def resolve_inaturalist_taxon(
    taxon_name: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    per_page: int = 30,
) -> ResolvedTaxon:
    """Resolve a scientific taxon name to exact iNaturalist taxon metadata."""
    name_clean = taxon_name.strip()
    if not name_clean:
        raise ValueError("taxon_name must be non-empty.")
    limit = min(max(per_page, 1), 30)
    params: dict[str, Any] = {"q": name_clean, "per_page": limit}
    payload = _fetch_json(INAT_TAXA_URL, params=params, user_agent=user_agent)
    needle = name_clean.casefold()
    for taxon in payload.get("results", []):
        if str(taxon.get("name", "")).strip().casefold() != needle:
            continue
        tid = taxon.get("id")
        if tid is None:
            continue
        return ResolvedTaxon(
            id=int(tid),
            name=str(taxon.get("name", "")).strip(),
            rank=str(taxon.get("rank", "")).strip(),
        )
    raise ValueError(
        f"No iNaturalist taxon with exact scientific name {taxon_name!r} in the first {limit} "
        "/v1/taxa search results. Check spelling or pass --taxon-id (lookup at "
        "https://www.inaturalist.org/taxa/search)."
    )


def resolve_inaturalist_taxon_id(
    taxon_name: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    per_page: int = 30,
) -> int:
    """Map a scientific name to iNaturalist's taxon id via /v1/taxa (exact name match).

    The observations API's ``taxon_name`` parameter is unreliable for some taxa (e.g. ``Mammalia``
    returns zero results; ``Anura`` can match a tiny wrong slice). Querying by ``taxon_id`` avoids
    that. See https://api.inaturalist.org/v1/docs/
    """
    return resolve_inaturalist_taxon(taxon_name, user_agent=user_agent, per_page=per_page).id


def collect_inaturalist_records(
    *,
    label: str,
    max_images: int,
    user_agent: str = DEFAULT_USER_AGENT,
    taxon_name: str | None = None,
    taxon_id: int | None = None,
    quality_grade: str = "research",
    per_page: int = 50,
    photo_size: str = "large",
    captive: str = "false",
    photo_license_allowlist: tuple[str, ...] = DEFAULT_PUBLIC_LICENSES,
    start_page: int = 1,
    max_pages: int | None = None,
    manifest_path: Path | None = None,
    flush_every_page: bool = False,
) -> CollectInaturalistResult:
    if max_images <= 0:
        raise ValueError("max_images must be positive.")
    if start_page < 1:
        raise ValueError("start_page must be >= 1.")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be >= 1 when provided.")
    if flush_every_page and manifest_path is None:
        raise ValueError("manifest_path is required when flush_every_page is True.")
    if not taxon_name and taxon_id is None:
        raise ValueError("Either taxon_name or taxon_id must be provided.")

    normalized_allowlist = {
        normalize_license_code(code)
        for code in photo_license_allowlist
        if normalize_license_code(code)
    }
    if taxon_id is not None:
        resolved_taxon = ResolvedTaxon(id=int(taxon_id), name="", rank="")
    else:
        assert taxon_name is not None
        resolved_taxon = resolve_inaturalist_taxon(taxon_name, user_agent=user_agent)
    resolved_id = resolved_taxon.id
    query_key = "taxon_id"
    query_value = str(resolved_id)
    page = start_page
    end_page_exclusive = (
        start_page + max_pages if max_pages is not None else _MAX_INAT_PAGES + 1
    )
    results: list[CollectedImageRecord] = []
    seen_photo_ids: set[str] = set()
    pages_fetched = 0
    manifest_written = 0
    manifest_skipped = 0
    dedupe_keys: set[tuple[str, str, str]] | None = None
    if flush_every_page and manifest_path is not None:
        dedupe_keys = load_manifest_keys(manifest_path) if manifest_path.exists() else set()

    while len(results) < max_images and page < end_page_exclusive and page <= _MAX_INAT_PAGES:
        if page > start_page:
            time.sleep(_INAT_OBSERVATIONS_PAGE_DELAY_SEC)
        params = {
            "page": page,
            "per_page": min(per_page, 200),
            "photos": "true",
            "quality_grade": quality_grade,
            "captive": captive,
            query_key: resolved_id,
        }
        payload = _fetch_json(INAT_OBSERVATIONS_URL, params=params, user_agent=user_agent)
        observations = payload.get("results", [])
        if not observations:
            break

        page_batch: list[CollectedImageRecord] = []
        for observation in observations:
            for photo in observation.get("photos", []):
                photo_id = str(photo.get("id", "")).strip()
                if not photo_id or photo_id in seen_photo_ids:
                    continue

                license_code = normalize_license_code(photo.get("license_code"))
                if normalized_allowlist and license_code not in normalized_allowlist:
                    continue

                image_url = _resolve_inat_photo_url(photo, preferred_size=photo_size)
                if not image_url:
                    continue

                seen_photo_ids.add(photo_id)
                taxon = observation.get("taxon") or {}
                latitude, longitude = _extract_latitude_longitude(observation)
                species_name = _extract_species_name(taxon)
                user = observation.get("user") or {}
                rec = CollectedImageRecord(
                    label=label,
                    provider="inaturalist",
                    query=query_key,
                    query_value=query_value,
                    resolved_taxon_id=str(resolved_taxon.id),
                    resolved_taxon_name=resolved_taxon.name,
                    resolved_rank=resolved_taxon.rank,
                    observation_id=str(observation.get("id", "")),
                    image_id=photo_id,
                    observation_taxon_id=_stringify_scalar(taxon.get("id")),
                    observation_taxon_name=str(taxon.get("name", "")),
                    observation_taxon_rank=str(taxon.get("rank", "")),
                    taxon_name=str(taxon.get("name", "")),
                    species_name=species_name,
                    quality_grade=str(observation.get("quality_grade", "")),
                    captive=_stringify_boolish(observation.get("captive")),
                    observed_on=str(observation.get("observed_on", "")),
                    observed_time=str(observation.get("time_observed_at", "")),
                    created_at=str(observation.get("created_at", "")),
                    updated_at=str(observation.get("updated_at", "")),
                    user_id=_stringify_scalar(user.get("id")),
                    user_login=str(user.get("login", "")),
                    latitude=latitude,
                    longitude=longitude,
                    positional_accuracy=_stringify_scalar(observation.get("positional_accuracy")),
                    place_guess=str(observation.get("place_guess", "")),
                    photo_license=license_code,
                    observation_license=normalize_license_code(observation.get("license_code")),
                    attribution=str(photo.get("attribution", "")),
                    image_url=image_url,
                    source_page_url=f"https://www.inaturalist.org/observations/{observation.get('id', '')}",
                    collected_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                results.append(rec)
                page_batch.append(rec)
                if len(results) >= max_images:
                    break
            if len(results) >= max_images:
                break

        if flush_every_page and manifest_path is not None and page_batch:
            use_append = manifest_path.exists()
            w, s = write_manifest(
                page_batch,
                manifest_path,
                append=use_append,
                dedupe=True,
                existing_keys=dedupe_keys,
            )
            manifest_written += w
            manifest_skipped += s

        page += 1
        pages_fetched += 1

    return CollectInaturalistResult(
        records=results,
        pages_fetched=pages_fetched,
        start_page=start_page,
        next_page=page,
        manifest_rows_written=manifest_written,
        manifest_rows_skipped_duplicate=manifest_skipped,
    )


def manifest_row_key(record: CollectedImageRecord) -> tuple[str, str, str]:
    return (record.label.strip(), record.provider.strip(), record.image_id.strip())


def load_manifest_keys(manifest_path: Path) -> set[tuple[str, str, str]]:
    if not manifest_path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("label") or "").strip()
            provider = (row.get("provider") or "").strip()
            image_id = (row.get("image_id") or "").strip()
            keys.add((label, provider, image_id))
    return keys


def write_manifest(
    records: list[CollectedImageRecord],
    manifest_path: Path,
    *,
    append: bool = False,
    dedupe: bool = False,
    existing_keys: set[tuple[str, str, str]] | None = None,
) -> tuple[int, int]:
    """Write manifest rows. Returns ``(written_count, skipped_duplicate_count)``."""
    ensure_parent(manifest_path)
    skipped = 0
    to_write = records
    if dedupe:
        if existing_keys is not None:
            keys = existing_keys
        else:
            keys = load_manifest_keys(manifest_path) if manifest_path.exists() else set()
        filtered: list[CollectedImageRecord] = []
        for record in records:
            key = manifest_row_key(record)
            if key in keys:
                skipped += 1
                continue
            keys.add(key)
            filtered.append(record)
        to_write = filtered

    if append and manifest_path.exists():
        _assert_manifest_header_matches_schema(manifest_path)
    mode = "a" if append and manifest_path.exists() else "w"
    write_header = mode == "w"
    written = 0
    try:
        handle = manifest_path.open(mode, newline="", encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot open manifest for writing (close Excel or other programs locking the file): {manifest_path}"
        ) from exc
    with handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        for record in to_write:
            writer.writerow(asdict(record))
            written += 1

    return written, skipped


def download_images_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    overwrite: bool = False,
) -> dict[str, int]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    downloaded = 0
    skipped_existing = 0
    failed = 0

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("label") or "unlabeled").strip() or "unlabeled"
            provider = (row.get("provider") or "source").strip() or "source"
            image_id = (row.get("image_id") or "image").strip() or "image"
            image_url = (row.get("image_url") or "").strip()
            if not image_url:
                failed += 1
                continue

            destination = output_dir / _slugify(label) / f"{provider}_{image_id}{_guess_extension(image_url)}"
            ensure_parent(destination)
            if destination.exists() and not overwrite:
                skipped_existing += 1
                continue

            try:
                content = _fetch_binary(image_url, user_agent=user_agent)
            except (HTTPError, URLError, TimeoutError, ValueError):
                failed += 1
                continue

            if not _is_valid_image_bytes(content):
                failed += 1
                continue

            destination.write_bytes(content)
            downloaded += 1

    return {
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }


def _fetch_json(url: str, *, params: dict[str, Any], user_agent: str) -> dict[str, Any]:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(
        request_url,
        headers=_inaturalist_request_headers(user_agent, accept="application/json"),
    )
    raw = _read_url_bytes_with_retry(request, timeout=_REQUEST_TIMEOUT_SEC)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"iNaturalist API response was not valid UTF-8 ({request_url})") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:800] if len(text) > 800 else text
        raise ValueError(
            f"iNaturalist API returned non-JSON (often HTML or an error page). URL: {request_url}. "
            f"Preview: {preview!r}"
        ) from exc


def _fetch_binary(url: str, *, user_agent: str) -> bytes:
    request = Request(
        url,
        headers=_inaturalist_request_headers(user_agent, accept="image/avif,image/webp,image/apng,image/*,*/*;q=0.8"),
    )
    content = _read_url_bytes_with_retry(request, timeout=_REQUEST_TIMEOUT_SEC)
    if not content:
        raise ValueError("Downloaded image was empty.")
    return content


def _resolve_inat_photo_url(photo: dict[str, Any], *, preferred_size: str) -> str:
    for key in (f"{preferred_size}_url", "original_url", "url"):
        value = str(photo.get(key, "")).strip()
        if value:
            if key == "url" and preferred_size != "square":
                return _swap_inat_size(value, preferred_size)
            return value
    return ""


def _swap_inat_size(url: str, size: str) -> str:
    for token in ("square", "thumb", "small", "medium", "large", "original"):
        marker = f"/{token}."
        if marker in url:
            return url.replace(marker, f"/{size}.")
    return url


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return normalized.strip("._") or "label"


def _guess_extension(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in sorted(RAW_IMAGE_EXTENSIONS, key=len, reverse=True):
        if path.endswith(ext):
            return ext
    return ".jpg"


def _assert_manifest_header_matches_schema(manifest_path: Path) -> None:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header != MANIFEST_FIELDS:
        raise ValueError(
            f"Manifest header mismatch for append: {manifest_path}. "
            "Write to a new manifest path when the schema changes."
        )


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _stringify_boolish(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return text
    return str(value)


def _extract_species_name(taxon: dict[str, Any]) -> str:
    if str(taxon.get("rank", "")).strip().lower() == "species":
        return str(taxon.get("name", ""))
    return ""


def _extract_latitude_longitude(observation: dict[str, Any]) -> tuple[str, str]:
    geojson = observation.get("geojson") or {}
    coordinates = geojson.get("coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
        return _stringify_scalar(coordinates[1]), _stringify_scalar(coordinates[0])

    location = str(observation.get("location", "")).strip()
    if location and "," in location:
        lat, lon = location.split(",", 1)
        return lat.strip(), lon.strip()
    return "", ""


def _is_valid_image_bytes(data: bytes) -> bool:
    """Return True if bytes decode as a raster image (rejects HTML/error bodies)."""
    if len(data) < 8:
        return False
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            fmt = (image.format or "").upper()
    except OSError:
        return False
    return fmt in {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "MPO"}


def _read_url_bytes_with_retry(request: Request, *, timeout: int) -> bytes:
    last_error: Exception | None = None
    attempts = len(_REQUEST_RETRY_DELAYS_SEC) + 1
    for attempt_idx in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRYABLE_HTTP_STATUS_CODES or attempt_idx >= attempts - 1:
                raise
        except (URLError, TimeoutError) as exc:
            last_error = exc
            if attempt_idx >= attempts - 1:
                raise

        time.sleep(_REQUEST_RETRY_DELAYS_SEC[attempt_idx])

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unreachable retry state in _read_url_bytes_with_retry")
