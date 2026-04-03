# Collection run log

Append commands here as you run them so the corpus stays **reproducible** and explainable.

**Naming for the first main milestone:** manifest `data/manifests/collection_main_v1.csv`, raw `data/raw_collection_main_v1`, split `data/split_collection_main_v1`, artifacts `artifacts/collection_main_v1/`.

**Targets (first main pass):** roughly **4k-5k** `snake`, **5k-7k** `no_snake`; negatives via **many taxa, smaller `--max-images`** each.

**Manifest snapshots:** use **`data/manifests/snapshots/`** (tracked empty folder + `.gitkeep`; snapshot `*.csv` files are **gitignored** by default). Copy the CSV **before** download (intended corpus) and optionally **after** download (failed URLs / filtered rows). Do not overwrite the pre-download snapshot.

**Disk:** `split` **duplicates** images (raw + split is about **2x** image bytes on disk). See [real_image_collection.md](real_image_collection.md#disk-space-split-duplicates-images) for size estimates and free-space guidance.

---

## collection_main_v1

### Freeze before the long run (recommended)

**Yes: finalize a budgeted negative mix before you start**, not taxon-by-taxon while collecting. Improvising mid-run skews the corpus toward whatever was easy to query last and breaks reproducibility.

Fill this section **before** the first sequential `collect-inat` campaign. Then run **only** the frozen list (plus snake), in order, with logging after each command.

#### Rules for v1 `no_snake` (example policy)

| Rule | Target |
|------|--------|
| Total `no_snake` rows | about **5,500** (adjust with snake target if needed) |
| Any single taxon | stay under about **20%** of the `no_snake` set |
| Insect-heavy buckets (`Lepidoptera` + `Odonata`) combined | stay under about **15%** of `no_snake` (reduces bug-heavy pilot bias) |
| Mix | include **close biological / shape confusers** (herps, fish, etc.), not only broad "other animals" |
| Lizard-like clades | prefer **`--taxon-id`** after manual verification; avoid guessing names |

#### Fallback if a taxon under-collects

If the API returns far fewer rows than `--max-images`:

- Do **not** top up by raising caps on an already-large taxon.
- Redistribute the shortfall across **2-3** other buckets that are still small relative to the 20% cap.

Document any change (taxon, old cap, new cap, reason) in this file.

---

### Frozen negative table (copy one plan, then edit resolution column)

**Column `resolution`:** write `name` or `id:<number>` and match the CLI flag you use.

#### Plan A (preferred): core mix 4,600 + 600-900 lizard-like via verified `--taxon-id`

Core totals **4,600**; add **600-900** in one or two verified lizard-like rows to reach about **5,200-5,500**.

| Taxon (scientific name) | `--max-images` | resolution | Notes |
|-------------------------|----------------|------------|-------|
| Mammalia | 900 | name | |
| Aves | 900 | name | |
| Anura | 700 | name | |
| Actinopterygii | 700 | name | |
| Testudines | 400 | name | |
| Caudata | 300 | name | |
| Crocodylia | 150 | name | |
| Lepidoptera | 250 | name | insect bucket (counts toward ~15% insect cap) |
| Odonata | 100 | name | insect bucket |
| Araneae | 200 | name | texture / legs; not counted in insect % above |
| *Lizard-like clade 1* | *TBD* | **id only** | fill after verification |
| *Lizard-like clade 2 (optional)* | *TBD* | **id only** | |

#### Plan B (no lizard pause): single table about **5,050** `no_snake`

Weaker on squamate-like negatives, but still **disciplined** and better than improvising.

| Taxon | `--max-images` | resolution |
|-------|----------------|------------|
| Mammalia | 1000 | name |
| Aves | 1000 | name |
| Anura | 800 | name |
| Actinopterygii | 800 | name |
| Testudines | 450 | name |
| Caudata | 300 | name |
| Crocodylia | 150 | name |
| Lepidoptera | 250 | name |
| Odonata | 100 | name |
| Araneae | 200 | name |

#### Snake row (same manifest; no `--append` on first command)

| Taxon | `--max-images` | resolution |
|-------|----------------|------------|
| Serpentes | *your target (e.g. 4500)* | name |

---

### Pre-download snapshot

- File: `data/manifests/snapshots/collection_main_v1_pre_download.csv` (or dated variant)
- Date:
- Total rows / snake / no_snake (after collection, before download):

### Commands (paste below, newest at bottom)

Run order should match the **frozen table** above (snake first without `--append`, then each `no_snake` line with `--append`).

If the iNaturalist API returns **403** after deep paging, use **chunked, flushable** snake pulls (see [real_image_collection.md](real_image_collection.md#resumable-collection-collect-inat)): repeat `collect-inat` with `--start-page`, `--max-pages`, `--flush-every-page`, and `--append` after the first chunk. Log each chunk's `next_page` from the JSON output for the next `--start-page`.

```text
# Example: first row creates the file (no --append):
# .venv\Scripts\python -m snake_detector.cli collect-inat --label snake --taxon-name Serpentes --manifest-path data/manifests/collection_main_v1.csv --max-images ...

# Example: append negatives (many small runs):
# .venv\Scripts\python -m snake_detector.cli collect-inat --label no_snake --taxon-name Anura --manifest-path data/manifests/collection_main_v1.csv --max-images ... --append

# Example: resumable snake chunk (repeat with increasing --start-page):
# .venv\Scripts\python -m snake_detector.cli collect-inat --label snake --taxon-name Serpentes --manifest-path data/manifests/collection_main_v1.csv --max-images 4500 --per-page 30 --start-page 1 --max-pages 20 --flush-every-page --user-agent "..."
# .venv\Scripts\python -m snake_detector.cli collect-inat ... --start-page 21 --max-pages 20 --flush-every-page --append --user-agent "..."
```

### Download

```text
# .venv\Scripts\python -m snake_detector.cli download-manifest --manifest-path data/manifests/collection_main_v1.csv --output-dir data/raw_collection_main_v1
```

### Post-download snapshot (optional)

- File:
- Notes (failures, manual drops):
