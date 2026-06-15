# Benchmarks

Criterion benchmarks for the hot paths in `datacube-core`. Run them with:

```bash
cargo bench -p datacube-core                  # all
cargo bench -p datacube-core --bench trend    # per-series statistics
cargo bench -p datacube-core --bench cube_ops # par_map_series + temporal ops
```

HTML reports and machine-readable estimates land in `target/criterion/`.

## Reference numbers

Indicative medians from one run on a 12th Gen Intel Core i7-1270P (16 logical
cores, laptop — expect thermal variability between runs), Rust 1.94.1,
`--release`. Absolute times are machine-dependent; the **shape** (scaling,
relative cost) is the point.

### Per-series statistics (`benches/trend.rs`)

NDVI-like series, ~23 obs/yr, ~10% cloud gaps. `n` = observations
(50 ≈ 2 yr Sentinel-2, 230 ≈ 6 yr at biweekly, 1000 ≈ dense/daily record).

| estimator         | n=50    | n=230   | n=1000  | scaling          |
|-------------------|---------|---------|---------|------------------|
| `linear_trend`    | 1.4 µs  | 3.8 µs  | 13 µs   | O(n)             |
| `mann_kendall`    | 3.4 µs  | 51 µs   | 1.0 ms  | O(n²) concordances|
| `theil_sen`       | 26 µs   | 120 µs  | 5.2 ms  | O(n²) pairwise   |
| `harmonic` (K=2)  | 13 µs   | 188 µs  | 440 µs  | O(n)             |
| `detect_breaks`   | 16 µs   | 234 µs  | 412 µs  | O(n²) per segment|

`theil_sen` and `mann_kendall` are O(n²) in the number of finite observations
(all pairwise slopes / concordances) and dominate long records — at n=1000
`theil_sen` is ~400× `linear_trend`. `linear_trend` and `harmonic` stay
linear and cheap.

### Cube operations (`benches/cube_ops.rs`)

`par_map_series` over `(1, ny, nx, 60)` cubes, Rayon across all cores.

| estimator      | 64×64 (4 k px) | 128×128 (16 k) | 256×256 (65 k) | throughput  |
|----------------|----------------|----------------|----------------|-------------|
| `theil_sen`    | 44 ms          | 116 ms         | 581 ms         | ~110 Kpx/s  |
| `mann_kendall` | 33 ms          | 70 ms          | 326 ms         | ~120–230 Kpx/s |
| `harmonic` K=2 | 51 ms          | 140 ms         | 808 ms         | ~80–120 Kpx/s |

Wall-clock grows roughly linearly with pixel count (throughput stays in the
same band as the grid grows 16×, 4 k → 65 k px), confirming `par_map_series`
scales with the Rayon pool — the per-pixel estimator cost, not parallel
overhead, sets the time. A full Sentinel-2 tile (≈10 980²) is ~17× a 256²
chunk, so a Theil-Sen + Mann-Kendall trend map of one tile/year is ~minutes
on this laptop, seconds on a many-core server.

Temporal transforms over a `(1, 128, 128, 120)` cube, Rayon-parallel over
pixels:

| op                           | time   | vs serial | note                  |
|------------------------------|--------|-----------|-----------------------|
| `composite` (monthly median) | 11 ms  | ~4× faster| compute-bound (median)|
| `gapfill_linear`             | 12 ms  | ~unchanged| bandwidth-bound       |

`composite` parallelizes well — the per-group median selection is real
per-pixel work. `gapfill_linear` barely moves: it does little arithmetic per
element and is dominated by cloning the cube (~16 MB here) and memory
traffic, so it is memory-bandwidth bound rather than CPU bound; the parallel
fill still helps on wider (multi-band) cubes where the copy amortizes.
