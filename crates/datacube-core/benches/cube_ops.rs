//! Cube-level operations: the Rayon-parallel `par_map_series` hot path and
//! the temporal transforms (composite, gap-fill), over cube sizes spanning
//! a small AOI tile up to a moderate scene chunk.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datacube_core::{CompositeMethod, CompositeWindow, Cube, stats};
use ndarray::Array4;

/// Builds a `(1, ny, nx, nt)` NDVI-like cube with ~10% cloud gaps.
fn cube(ny: usize, nx: usize, nt: usize) -> Cube {
    let mut data = Array4::<f64>::zeros((1, ny, nx, nt));
    for ((_, y, x, t), v) in data.indexed_iter_mut() {
        let i = t + nt * (x + nx * y);
        if i % 10 == 7 {
            *v = f64::NAN;
            continue;
        }
        let tt = t as f64 / 23.0;
        let noise = (i as f64 * 12.9898).sin() * 43758.5453;
        *v = 0.4
            + 0.01 * tt
            + 0.2 * (2.0 * std::f64::consts::PI * tt).cos()
            + 0.03 * (noise - noise.round());
    }
    // ascending, distinct times so composite/gapfill have well-defined axes
    let times: Vec<f64> = (0..nt).map(|t| t as f64 / 23.0).collect();
    Cube::new(data, times, vec!["ndvi".into()]).unwrap()
}

fn bench_par_map_series(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_map_series");
    // (ny, nx, nt): 64², 128² and 256² pixels over a 5-year monthly record
    for &(ny, nx, nt) in &[(64, 64, 60), (128, 128, 60), (256, 256, 60)] {
        let cube = cube(ny, nx, nt);
        let px = (ny * nx) as u64;
        group.throughput(Throughput::Elements(px));
        let label = format!("{ny}x{nx}x{nt}");

        group.bench_with_input(BenchmarkId::new("theil_sen", &label), &cube, |b, cube| {
            b.iter(|| {
                cube.par_map_series(0, |t, y| {
                    stats::theil_sen(t, y).map(|r| r.slope).unwrap_or(f64::NAN)
                })
                .unwrap()
            });
        });
        group.bench_with_input(
            BenchmarkId::new("mann_kendall", &label),
            &cube,
            |b, cube| {
                b.iter(|| {
                    cube.par_map_series(0, |_t, y| {
                        stats::mann_kendall(y)
                            .map(|r| r.p_value)
                            .unwrap_or(f64::NAN)
                    })
                    .unwrap()
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("harmonic_2", &label), &cube, |b, cube| {
            b.iter(|| {
                cube.par_map_series(0, |t, y| {
                    stats::harmonic_regression(t, y, 1.0, 2)
                        .map(|r| r.slope)
                        .unwrap_or(f64::NAN)
                })
                .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_temporal(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal");
    let cube = cube(128, 128, 120); // 5-year ~biweekly record
    group.throughput(Throughput::Elements((128 * 128) as u64));

    group.bench_function("composite_monthly_median", |b| {
        b.iter(|| {
            black_box(&cube)
                .composite(CompositeWindow::Period(1.0 / 12.0), CompositeMethod::Median)
                .unwrap()
        });
    });
    group.bench_function("gapfill_linear", |b| {
        b.iter(|| black_box(&cube).gapfill_linear(Some(0.25)).unwrap());
    });
    group.finish();
}

criterion_group!(benches, bench_par_map_series, bench_temporal);
criterion_main!(benches);
