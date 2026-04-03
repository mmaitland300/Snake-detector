# Real Wildlife Image Collection

The current checked-in `model.joblib` is a placeholder-safe demo artifact, not a good wildlife-photo model.

If you want to retrain this project on real snake photography, the safest repeatable path is:

1. Build a metadata manifest from a source with an API and explicit license fields.
2. Download only records that meet your license policy.
3. Keep the real-photo corpus local unless redistribution rights are fully documented.
4. Retrain and re-evaluate before deciding whether to host a public demo.

## Why not return to Google Images scraping?

- This repo already documents that the historical scraped corpus has unverified rights.
- GBIF's API documentation explicitly says scripted scraping of `www.gbif.org` is unsupported and may be blocked; use APIs instead.
- iNaturalist exposes an observations API with photo and license filters, which is a better fit for reproducible collection.

## Taxon names vs ids

`collect-inat` resolves `--taxon-name` with iNaturalist's **`/v1/taxa`** search and uses an **exact scientific-name match** in the first page of results, then queries observations by **`taxon_id`**. That avoids API quirks where **`taxon_name` on `/v1/observations` returns no rows** for some taxa (for example **Mammalia**, **Reptilia**) or a **tiny wrong slice** for others (for example **Anura**). You can still pass **`--taxon-id`** to skip the taxa lookup.

## New CLI flow

Build a snake manifest from iNaturalist:

```bash
.venv\Scripts\python -m snake_detector.cli collect-inat --label snake --taxon-name Serpentes --manifest-path data/manifests/real_snakes.csv --max-images 400
```

Append a non-snake class from another taxon:

```bash
.venv\Scripts\python -m snake_detector.cli collect-inat --label no_snake --taxon-name Anura --manifest-path data/manifests/real_snakes.csv --max-images 400 --append
```

Download the manifest into raw class folders:

```bash
.venv\Scripts\python -m snake_detector.cli download-manifest --manifest-path data/manifests/real_snakes.csv --output-dir data/raw_real
```

Then reuse the existing split/train/eval commands against `data/raw_real`.

## Resumable collection (`collect-inat`)

Long Serpentes (or other) pulls can hit CDN edge limits (HTTP 403) after many pages. By default the CLI used to write the manifest only after the whole run finished, so a late failure could discard everything.

These flags make collection **checkpointed** and **resumable**:

| Flag | Role |
|------|------|
| `--start-page N` | First observations API page for this run (1-based). |
| `--max-pages M` | Fetch at most `M` pages this run, then stop (unless `--max-images` is hit first). |
| `--flush-every-page` | After each successful API page, append new rows to the manifest (deduped). Survives mid-run crashes or 403s after earlier pages were written. |
| `--append` | Append to an existing manifest. **Required** with `--flush-every-page` if the manifest file already exists (the CLI refuses to truncate an existing file when flushing). |

**Dedupe:** When appending, rows matching an existing `(label, provider, image_id)` are skipped so reruns and overlapping chunks do not duplicate lines.

**JSON summary** printed at the end includes `records_collected`, `records_written`, `records_skipped_duplicate`, `start_page`, `next_page`, and `pages_fetched`. Use `next_page` as the next `--start-page` when resuming a windowed pull.

The Phase 1 helper script `scripts/collection_main_v1_01_collect.ps1` defaults to **chunked, flushed** Serpentes collection: it stops when the manifest has at least `$SnakeTargetRows` snake rows, when a chunk fetches zero pages, or when `$SnakeMaxChunks` is hit (raise that cap if you stop short of the target). Set `$UseSnakeChunks = $false` only if you want a single non-checkpointed `collect-inat` run.

**Example: chunked snake pull** (adjust windows until you reach your `--max-images` target or the API returns empty pages):

```text
# Chunk 1 (creates the CSV)
python -m snake_detector.cli collect-inat --label snake --taxon-name Serpentes --manifest-path data/manifests/collection_main_v1.csv --max-images 4500 --per-page 30 --start-page 1 --max-pages 20 --flush-every-page --user-agent "YOUR_UA"

# Chunk 2 (append; same max-images cap per invocation)
python -m snake_detector.cli collect-inat --label snake --taxon-name Serpentes --manifest-path data/manifests/collection_main_v1.csv --max-images 4500 --per-page 30 --start-page 21 --max-pages 20 --flush-every-page --append --user-agent "YOUR_UA"
```

Then continue negatives with `--append` as before (dedupe still applies).

## First "main" collection (modest scale, auditable)

For the **first** large pull after pilots, stay **big enough to matter** but **small enough to spot problems** in the manifest and a sample of images:

| Side | Suggested row count (order of magnitude) |
|------|------------------------------------------|
| `snake` | about **4k-5k** |
| `no_snake` | about **5k-7k** |

Avoid jumping straight to very large targets (for example 8k / 10k) until this pass looks clean.

### Fragmented negatives (before per-species / per-observer caps)

Prefer **many** negative taxa with **smaller** `--max-images` each, instead of a few huge pulls. That limits **taxon dominance** until you add automatic caps on species or observer.

### Naming (one milestone = one suffix)

Use a consistent stem so paths stay unambiguous later:

| Artifact | Example path |
|----------|----------------|
| Manifest | `data/manifests/collection_main_v1.csv` |
| Raw images | `data/raw_collection_main_v1/` |
| Split output | `data/split_collection_main_v1/` |
| Training artifacts | `artifacts/collection_main_v1/` |

### Command history (reproducibility)

Append every `collect-inat` (and related) command to a log in the repo, for example:

- [docs/collection_runs.md](collection_runs.md) (narrative + pasted commands), or
- `data/manifests/collection_main_v1_commands.txt` next to the manifest.

Include date, taxon, label, `--max-images`, and whether `--append` was used.

### Manifest snapshots

The repo includes an empty **`data/manifests/snapshots/`** folder (via `.gitkeep`) as the intentional place for snapshot copies. **Generated snapshot CSVs are gitignored by default** so the repo does not bloat; you can still `git add -f` a specific file if you decide one version should live in history.

1. **Before download:** copy the manifest here (or beside the live manifest) as the **source of truth** for what you *intended* to collect, e.g. `collection_main_v1_pre_download.csv` or `collection_main_v1_YYYYMMDD_pre_download.csv`.
2. **After download:** save another copy if you later annotate **failed URLs**, dropped rows, or post-filters (e.g. `collection_main_v1_post_download.csv`), without overwriting the pre-download file.

### Disk space (split duplicates images)

The **`split` step copies** every image into train/val/test (same bytes as raw). Pilot v2 showed roughly **~489 KB per image** on average; raw and split trees were the **same file count and total size** (for example ~1,020 files and ~487 MB each).

Rough planning numbers:

| Total images (order of magnitude) | Raw only | + split copy (~2x images on disk) |
|-----------------------------------|----------|-------------------------------------|
| ~9,000 | ~4.3 GB | ~8.6 GB |
| ~12,000 | ~5.7 GB | ~11.4 GB |

Add margin for **failed/retried downloads**, manifests, logs, model artifacts, and any **extra collections or holdout copies**. Practical targets: **at least ~15 GB free** on the drive that holds the project; **~20 GB** is more comfortable; plan for more if you expect multiple milestones or lockbox trees.

**Manual free-space check (Windows):** In File Explorer, right-click the drive, open **Properties**; or in PowerShell: `Get-Volume C` (adjust drive letter if needed).

**Future engineering note:** if disk gets tight, a follow-up improvement is **non-copy splits** (symlinks, hardlinks, or manifest-based loaders). Until then, budget for **near-2x** image storage (raw + split).

### Workflow summary

1. Create `collection_main_v1.csv` with the first `collect-inat` (no `--append`).
2. Append many smaller negative (and any extra snake) runs; log commands.
3. Snapshot manifest to a **pre-download** copy.
4. Run `download-manifest` into `data/raw_collection_main_v1` (re-runs skip existing files).
5. Optional: snapshot to a **post-download** copy if you track failures or edits.
6. Run `split` with `--manifest-path` and grouped `--group-by` into `data/split_collection_main_v1`.
7. Train/eval with outputs under `artifacts/collection_main_v1/`.

Reserve a **later milestone** (lockbox / final holdout) before any **public** benchmark claim; see project README for the split/manifest story.

## Practical label strategy

For a better `snake` vs `no_snake` detector, do not make the negative class only blank backgrounds.
Include animals and scenes that are commonly confused with snakes:

- lizards
- eels
- ropes / hoses / vines
- leaf litter / branches / rocks
- close-up ground vegetation

## Hosting recommendation

Do not market or host the current placeholder model as a real wildlife identifier.
Either:

- keep the current demo as an engineering prototype with explicit placeholder-data language, or
- replace it only after retraining on real photos and validating on a held-out real-photo test set.
