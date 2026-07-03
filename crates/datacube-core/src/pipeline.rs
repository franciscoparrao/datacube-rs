//! Chunked execution of the composite → gapfill → index → trend/breaks
//! chain (H5 step 3 of the roadmap audit).
//!
//! Each stage is a per-pixel-series transform ([`Cube::composite`],
//! [`Cube::gapfill_linear`], the [`crate::indices`], and [`Cube::par_map_series`]
//! together with [`crate::stats`]): no stage reads a neighboring pixel, so
//! running the whole chain independently on spatial tiles produces exactly
//! the same result as running it on the full cube — only the peak memory
//! changes. [`Cube::run_chunked`] bounds that peak to one chunk's cube (plus
//! whatever its own stages allocate) instead of one full-cube copy per stage.

use ndarray::Array2;

use crate::cube::Cube;
use crate::error::CubeError;
use crate::indices;
use crate::stats::{self, BreakOptions};
use crate::temporal::{CompositeMethod, CompositeWindow, composite_time_axis};

/// Linear gap-filling parameters for a [`ChunkPipeline`]; see
/// [`Cube::gapfill_linear`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GapfillSpec {
    /// Gaps wider than this (in time units) are left as `NaN`; `None` = no limit.
    pub max_gap: Option<f64>,
}

/// A spectral index to derive from named bands before analysis. Thin
/// wrappers over [`crate::indices`], keyed by asset/band name so the pipeline
/// can be built from CLI/user-facing band roles.
#[derive(Debug, Clone)]
pub enum IndexSpec {
    Ndvi {
        nir: String,
        red: String,
    },
    Ndwi {
        green: String,
        nir: String,
    },
    Nbr {
        nir: String,
        swir: String,
    },
    Ndbi {
        swir: String,
        nir: String,
    },
    Evi {
        nir: String,
        red: String,
        blue: String,
    },
    Savi {
        nir: String,
        red: String,
        l: f64,
    },
}

impl IndexSpec {
    /// The single-band label the derived cube carries (`cube.bands()[0]`
    /// after [`IndexSpec::apply`]) — matches the fixed labels used by
    /// [`crate::indices`].
    pub fn label(&self) -> &'static str {
        match self {
            IndexSpec::Ndvi { .. } => "ndvi",
            IndexSpec::Ndwi { .. } => "ndwi",
            IndexSpec::Nbr { .. } => "nbr",
            IndexSpec::Ndbi { .. } => "ndbi",
            IndexSpec::Evi { .. } => "evi",
            IndexSpec::Savi { .. } => "savi",
        }
    }

    fn apply(&self, cube: &Cube) -> Result<Cube, CubeError> {
        match self {
            IndexSpec::Ndvi { nir, red } => indices::ndvi(cube, nir, red),
            IndexSpec::Ndwi { green, nir } => indices::ndwi(cube, green, nir),
            IndexSpec::Nbr { nir, swir } => indices::nbr(cube, nir, swir),
            IndexSpec::Ndbi { swir, nir } => indices::ndbi(cube, swir, nir),
            IndexSpec::Evi { nir, red, blue } => indices::evi(cube, nir, red, blue),
            IndexSpec::Savi { nir, red, l } => indices::savi(cube, nir, red, *l),
        }
    }
}

/// Per-pixel trend estimator for [`StatSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendMethod {
    /// Theil-Sen slope + Mann-Kendall p-value (robust to outliers/irregular sampling).
    TheilSenMannKendall,
    /// Ordinary least squares slope + t-test p-value.
    Ols,
}

/// Which per-pixel statistics to compute on `band`, evaluated after
/// composite/gapfill/index have been applied.
#[derive(Debug, Clone)]
pub struct StatSpec {
    /// Band name to analyze, resolved against the cube *after* composite/
    /// gapfill/index — so this is an index label (e.g. `"ndvi"`) when
    /// [`ChunkPipeline::index`] is set, or an original asset key otherwise.
    pub band: String,
    pub trend: Option<TrendMethod>,
    pub breaks: Option<BreakOptions>,
}

/// Per-pixel statistic grids for one chunk, matching the `Some` fields of
/// the [`StatSpec`] that produced them.
#[derive(Debug, Clone)]
pub struct ChunkStat {
    /// `(slope, p_value)`.
    pub trend: Option<(Array2<f64>, Array2<f64>)>,
    /// `(break_count, first_break_time)`.
    pub breaks: Option<(Array2<f64>, Array2<f64>)>,
}

/// The composite → gapfill → index → stat chain run per spatial chunk by
/// [`Cube::run_chunked`].
#[derive(Debug, Clone)]
pub struct ChunkPipeline {
    pub composite: Option<(CompositeWindow, CompositeMethod)>,
    pub gapfill: Option<GapfillSpec>,
    pub index: Option<IndexSpec>,
    pub stat: StatSpec,
}

/// One chunk's result, positioned by the `(y0, x0)` convention of
/// [`Cube::chunks`] (top-left corner within the parent cube).
#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub y0: usize,
    pub x0: usize,
    pub stat: ChunkStat,
}

impl ChunkPipeline {
    /// The time coordinates the cube would carry after `composite`, without
    /// touching any pixel data — composite/gapfill/index never change the
    /// spatial extent, so this plus the original `(height, width)` fully
    /// describe the pipeline's output shape without running it.
    pub fn output_time(&self, cube: &Cube) -> Result<Vec<f64>, CubeError> {
        match self.composite {
            Some((window, _)) => composite_time_axis(cube.time(), window),
            None => Ok(cube.time().to_vec()),
        }
    }

    fn run_on(&self, cube: &Cube) -> Result<ChunkStat, CubeError> {
        let mut cube = match self.composite {
            Some((window, method)) => cube.composite(window, method)?,
            None => cube.clone(),
        };
        if let Some(gf) = &self.gapfill {
            cube = cube.gapfill_linear(gf.max_gap)?;
        }
        if let Some(index) = &self.index {
            cube = index.apply(&cube)?;
        }
        let band = cube.band(&self.stat.band)?;

        let trend = self
            .stat
            .trend
            .map(|method| {
                let out = cube.par_map_series(band, |t, y| match method {
                    TrendMethod::TheilSenMannKendall => {
                        let slope = stats::theil_sen(t, y).map(|r| r.slope).unwrap_or(f64::NAN);
                        let p = stats::mann_kendall(y)
                            .map(|r| r.p_value)
                            .unwrap_or(f64::NAN);
                        (slope, p)
                    }
                    TrendMethod::Ols => stats::linear_trend(t, y)
                        .map(|r| (r.slope, r.p_value))
                        .unwrap_or((f64::NAN, f64::NAN)),
                })?;
                Ok::<_, CubeError>((out.mapv(|(s, _)| s), out.mapv(|(_, p)| p)))
            })
            .transpose()?;

        let breaks = self
            .stat
            .breaks
            .as_ref()
            .map(|opts| {
                let out =
                    cube.par_map_series(band, |t, y| match stats::detect_breaks(t, y, opts) {
                        Ok(r) => {
                            let first = r.breaks.first().map(|b| b.time).unwrap_or(f64::NAN);
                            (r.breaks.len() as f64, first)
                        }
                        Err(_) => (f64::NAN, f64::NAN),
                    })?;
                Ok::<_, CubeError>((out.mapv(|(c, _)| c), out.mapv(|(_, f)| f)))
            })
            .transpose()?;

        Ok(ChunkStat { trend, breaks })
    }
}

impl Cube {
    /// Runs `pipeline` independently on every spatial tile of at most
    /// `chunk_y` × `chunk_x` pixels (see [`Cube::chunks`]), bounding peak
    /// memory to one chunk's cube instead of one full-cube copy per pipeline
    /// stage. Sequential across chunks; parallelism within a chunk still
    /// comes from [`Cube::par_map_series`]/band-math (Rayon).
    pub fn run_chunked<'a>(
        &'a self,
        chunk_y: usize,
        chunk_x: usize,
        pipeline: &'a ChunkPipeline,
    ) -> Result<impl Iterator<Item = Result<ChunkResult, CubeError>> + 'a, CubeError> {
        Ok(self.chunks(chunk_y, chunk_x)?.map(move |chunk| {
            let sub = Cube::new(
                chunk.data.to_owned(),
                self.time().to_vec(),
                self.bands().to_vec(),
            )?;
            let stat = pipeline.run_on(&sub)?;
            Ok(ChunkResult {
                y0: chunk.y0,
                x0: chunk.x0,
                stat,
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    /// value = t * (1 + y + x), with a NaN at (y=0,x=0,t=1) to exercise
    /// composite/gapfill NaN handling; nir/red bands set up so ndvi is
    /// well-defined everywhere it's finite.
    fn scene_cube(ny: usize, nx: usize, nt: usize) -> Cube {
        let mut nir = Array4::zeros((1, ny, nx, nt));
        for ((_, y, x, t), v) in nir.indexed_iter_mut() {
            *v = 10.0 + t as f64 * (1.0 + y as f64 + x as f64);
        }
        let mut red = Array4::zeros((1, ny, nx, nt));
        for ((_, y, x, t), v) in red.indexed_iter_mut() {
            *v = 5.0 + 0.5 * t as f64 * (1.0 + y as f64 + x as f64);
        }
        nir[[0, 0, 0, 1]] = f64::NAN;
        red[[0, 0, 0, 1]] = f64::NAN;

        let mut data = Array4::zeros((2, ny, nx, nt));
        data.slice_mut(ndarray::s![0, .., .., ..])
            .assign(&nir.slice(ndarray::s![0, .., .., ..]));
        data.slice_mut(ndarray::s![1, .., .., ..])
            .assign(&red.slice(ndarray::s![0, .., .., ..]));
        Cube::new(
            data,
            (0..nt).map(|t| t as f64).collect(),
            vec!["nir".into(), "red".into()],
        )
        .unwrap()
    }

    fn pipeline() -> ChunkPipeline {
        ChunkPipeline {
            composite: None,
            gapfill: Some(GapfillSpec { max_gap: None }),
            index: Some(IndexSpec::Ndvi {
                nir: "nir".into(),
                red: "red".into(),
            }),
            stat: StatSpec {
                band: "ndvi".into(),
                trend: Some(TrendMethod::TheilSenMannKendall),
                breaks: Some(BreakOptions::default()),
            },
        }
    }

    /// Reassembles per-chunk results into full-size grids so they can be
    /// compared against the whole-cube path.
    fn run_chunked_full(cube: &Cube, cy: usize, cx: usize, p: &ChunkPipeline) -> ChunkStatFull {
        let (_, ny, nx, _) = cube.dims();
        let mut slope = Array2::from_elem((ny, nx), f64::NAN);
        let mut pvalue = Array2::from_elem((ny, nx), f64::NAN);
        let mut count = Array2::from_elem((ny, nx), f64::NAN);
        let mut first = Array2::from_elem((ny, nx), f64::NAN);
        for result in cube.run_chunked(cy, cx, p).unwrap() {
            let ChunkResult { y0, x0, stat } = result.unwrap();
            if let Some((s, pv)) = &stat.trend {
                let (ch, cw) = s.dim();
                slope
                    .slice_mut(ndarray::s![y0..y0 + ch, x0..x0 + cw])
                    .assign(s);
                pvalue
                    .slice_mut(ndarray::s![y0..y0 + ch, x0..x0 + cw])
                    .assign(pv);
            }
            if let Some((c, f)) = &stat.breaks {
                let (ch, cw) = c.dim();
                count
                    .slice_mut(ndarray::s![y0..y0 + ch, x0..x0 + cw])
                    .assign(c);
                first
                    .slice_mut(ndarray::s![y0..y0 + ch, x0..x0 + cw])
                    .assign(f);
            }
        }
        ChunkStatFull {
            slope,
            pvalue,
            count,
            first,
        }
    }

    struct ChunkStatFull {
        slope: Array2<f64>,
        pvalue: Array2<f64>,
        count: Array2<f64>,
        first: Array2<f64>,
    }

    fn run_whole_cube(cube: &Cube, p: &ChunkPipeline) -> ChunkStatFull {
        let stat = p.run_on(cube).unwrap();
        let (slope, pvalue) = stat.trend.unwrap();
        let (count, first) = stat.breaks.unwrap();
        ChunkStatFull {
            slope,
            pvalue,
            count,
            first,
        }
    }

    fn assert_grids_eq(a: &ChunkStatFull, b: &ChunkStatFull) {
        for (g1, g2) in [
            (&a.slope, &b.slope),
            (&a.pvalue, &b.pvalue),
            (&a.count, &b.count),
            (&a.first, &b.first),
        ] {
            for (l, r) in g1.iter().zip(g2.iter()) {
                assert!(
                    l.is_nan() == r.is_nan() && (l.is_nan() || l == r),
                    "{l} != {r}"
                );
            }
        }
    }

    #[test]
    fn chunked_matches_whole_cube_exactly() {
        let cube = scene_cube(7, 5, 12);
        let p = pipeline();
        let chunked = run_chunked_full(&cube, 2, 3, &p);
        let whole = run_whole_cube(&cube, &p);
        assert_grids_eq(&chunked, &whole);
    }

    #[test]
    fn chunked_matches_whole_cube_with_composite() {
        let mut p = pipeline();
        p.composite = Some((CompositeWindow::Period(2.0), CompositeMethod::Median));
        let cube = scene_cube(5, 4, 20);
        let chunked = run_chunked_full(&cube, 2, 2, &p);
        let whole = run_whole_cube(&cube, &p);
        assert_grids_eq(&chunked, &whole);
    }

    #[test]
    fn non_divisor_chunk_size_covers_every_pixel() {
        let cube = scene_cube(7, 5, 12);
        let p = pipeline();
        let mut seen = Array2::from_elem((7, 5), false);
        for result in cube.run_chunked(3, 4, &p).unwrap() {
            let ChunkResult { y0, x0, stat } = result.unwrap();
            let (ch, cw) = stat.trend.as_ref().unwrap().0.dim();
            seen.slice_mut(ndarray::s![y0..y0 + ch, x0..x0 + cw])
                .fill(true);
        }
        assert!(seen.iter().all(|&v| v));
    }

    #[test]
    fn output_time_matches_composite_without_running_pipeline() {
        let cube = scene_cube(2, 2, 6);
        let window = CompositeWindow::Period(2.0);
        let p = ChunkPipeline {
            composite: Some((window, CompositeMethod::Mean)),
            gapfill: None,
            index: None,
            stat: StatSpec {
                band: "nir".into(),
                trend: None,
                breaks: None,
            },
        };
        let expected = cube.composite(window, CompositeMethod::Mean).unwrap();
        assert_eq!(p.output_time(&cube).unwrap(), expected.time());
    }

    #[test]
    fn output_time_without_composite_is_original_time() {
        let cube = scene_cube(2, 2, 4);
        let p = ChunkPipeline {
            composite: None,
            gapfill: None,
            index: None,
            stat: StatSpec {
                band: "nir".into(),
                trend: None,
                breaks: None,
            },
        };
        assert_eq!(p.output_time(&cube).unwrap(), cube.time());
    }

    #[test]
    fn invalid_band_name_propagates_from_first_chunk() {
        let cube = scene_cube(2, 2, 4);
        let mut p = pipeline();
        p.stat.band = "nope".into();
        p.index = None;
        let mut iter = cube.run_chunked(1, 1, &p).unwrap();
        assert!(matches!(
            iter.next().unwrap(),
            Err(CubeError::BandNotFound(_))
        ));
    }
}
