//! Per-series statistics, across series lengths typical of Landsat/Sentinel
//! archives (≈40 obs/yr Sentinel-2; 230 over a 6-year monthly record).

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datacube_core::stats::{self, BreakOptions};

/// A deterministic NDVI-like series: trend + annual cycle + pseudo-noise,
/// with ~10% of observations missing (cloud gaps).
fn series(n: usize) -> (Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..n).map(|i| i as f64 / 23.0).collect(); // ~23 obs/yr
    let y: Vec<f64> = t
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            if i % 10 == 7 {
                return f64::NAN; // cloud gap
            }
            let noise = (i as f64 * 12.9898).sin() * 43758.5453;
            0.4 + 0.01 * t
                + 0.2 * (2.0 * std::f64::consts::PI * t).cos()
                + 0.03 * (noise - noise.round())
        })
        .collect();
    (t, y)
}

fn bench_trend(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_series");
    for &n in &[50usize, 230, 1000] {
        let (t, y) = series(n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("linear_trend", n), &n, |b, _| {
            b.iter(|| stats::linear_trend(black_box(&t), black_box(&y)));
        });
        group.bench_with_input(BenchmarkId::new("theil_sen", n), &n, |b, _| {
            b.iter(|| stats::theil_sen(black_box(&t), black_box(&y)));
        });
        group.bench_with_input(BenchmarkId::new("mann_kendall", n), &n, |b, _| {
            b.iter(|| stats::mann_kendall(black_box(&y)));
        });
        group.bench_with_input(BenchmarkId::new("harmonic_2", n), &n, |b, _| {
            b.iter(|| stats::harmonic_regression(black_box(&t), black_box(&y), 1.0, 2));
        });
        group.bench_with_input(BenchmarkId::new("detect_breaks_h1", n), &n, |b, _| {
            let opts = BreakOptions {
                n_harmonics: 1,
                ..BreakOptions::default()
            };
            b.iter(|| stats::detect_breaks(black_box(&t), black_box(&y), &opts));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_trend);
criterion_main!(benches);
