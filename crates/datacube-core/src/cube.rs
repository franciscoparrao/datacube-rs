use ndarray::{Array2, Array4, ArrayView1, ArrayView4, s};
use rayon::prelude::*;

use crate::error::CubeError;

/// An in-memory temporal data cube with axes `(band, y, x, time)`.
///
/// The time axis is innermost so every per-pixel series is a contiguous
/// slice; iteration over pixels streams views without copying. Missing
/// observations are represented as `NaN` and are filtered by the
/// statistics in [`crate::stats`].
#[derive(Debug, Clone)]
pub struct Cube {
    data: Array4<f64>,
    time: Vec<f64>,
    bands: Vec<String>,
}

/// One pixel's time series, yielded by [`Cube::iter_series`].
#[derive(Debug)]
pub struct PixelSeries<'a> {
    pub y: usize,
    pub x: usize,
    pub values: ArrayView1<'a, f64>,
}

/// A spatial tile of the cube (all bands and times), yielded by [`Cube::chunks`].
#[derive(Debug)]
pub struct CubeChunk<'a> {
    /// Row offset of this chunk within the parent cube.
    pub y0: usize,
    /// Column offset of this chunk within the parent cube.
    pub x0: usize,
    pub data: ArrayView4<'a, f64>,
}

impl Cube {
    /// Builds a cube from a `(band, y, x, time)` array, time coordinates and
    /// band labels.
    ///
    /// `time` is a numeric coordinate (e.g. fractional years or epoch days);
    /// its length must equal the last axis of `data`, and `bands.len()` must
    /// equal the first axis. Non-standard-layout arrays are copied into
    /// standard layout so series stay contiguous.
    pub fn new(data: Array4<f64>, time: Vec<f64>, bands: Vec<String>) -> Result<Self, CubeError> {
        let (nb, _ny, _nx, nt) = data.dim();
        if time.len() != nt {
            return Err(CubeError::DimensionMismatch(format!(
                "time axis has {nt} steps but {} time coordinates were given",
                time.len()
            )));
        }
        if bands.len() != nb {
            return Err(CubeError::DimensionMismatch(format!(
                "data has {nb} bands but {} band labels were given",
                bands.len()
            )));
        }
        let data = if data.is_standard_layout() {
            data
        } else {
            data.as_standard_layout().to_owned()
        };
        Ok(Self { data, time, bands })
    }

    /// `(bands, height, width, time steps)`.
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        self.data.dim()
    }

    /// Time coordinates of the cube.
    pub fn time(&self) -> &[f64] {
        &self.time
    }

    /// Band labels.
    pub fn bands(&self) -> &[String] {
        self.bands.iter().as_slice()
    }

    /// Raw data view.
    pub fn data(&self) -> ArrayView4<'_, f64> {
        self.data.view()
    }

    /// The time series of one pixel in one band.
    pub fn series(
        &self,
        band: usize,
        y: usize,
        x: usize,
    ) -> Result<ArrayView1<'_, f64>, CubeError> {
        self.check_band(band)?;
        Ok(self.data.slice(s![band, y, x, ..]))
    }

    /// Streams every pixel series of `band` in row-major order.
    pub fn iter_series(
        &self,
        band: usize,
    ) -> Result<impl Iterator<Item = PixelSeries<'_>> + '_, CubeError> {
        self.check_band(band)?;
        let (_, ny, nx, _) = self.dims();
        let data = &self.data;
        Ok((0..ny * nx).map(move |i| {
            let (y, x) = (i / nx, i % nx);
            PixelSeries {
                y,
                x,
                values: data.slice(s![band, y, x, ..]),
            }
        }))
    }

    /// Applies `f(time, values)` to every pixel series of `band` in parallel
    /// (Rayon) and collects the results into a `(y, x)` grid.
    ///
    /// This is the core streaming primitive for per-pixel trend analysis.
    pub fn par_map_series<T, F>(&self, band: usize, f: F) -> Result<Array2<T>, CubeError>
    where
        T: Send,
        F: Fn(&[f64], &[f64]) -> T + Sync,
    {
        self.check_band(band)?;
        let (_, ny, nx, _) = self.dims();
        let time = self.time.as_slice();
        let mut out = Vec::with_capacity(ny * nx);
        (0..ny * nx)
            .into_par_iter()
            .map(|i| {
                let (y, x) = (i / nx, i % nx);
                let series = self.data.slice(s![band, y, x, ..]);
                match series.as_slice() {
                    Some(values) => f(time, values),
                    // unreachable for standard layout, kept as a safe fallback
                    None => f(time, &series.to_vec()),
                }
            })
            .collect_into_vec(&mut out);
        Array2::from_shape_vec((ny, nx), out)
            .map_err(|e| CubeError::DimensionMismatch(e.to_string()))
    }

    /// Iterates the cube as spatial tiles of at most `chunk_y` × `chunk_x`
    /// pixels (edge tiles may be smaller), keeping all bands and times.
    pub fn chunks(
        &self,
        chunk_y: usize,
        chunk_x: usize,
    ) -> Result<impl Iterator<Item = CubeChunk<'_>> + '_, CubeError> {
        if chunk_y == 0 || chunk_x == 0 {
            return Err(CubeError::InvalidChunkSize(format!(
                "chunk sizes must be > 0, got ({chunk_y}, {chunk_x})"
            )));
        }
        let (_, ny, nx, _) = self.dims();
        let data = &self.data;
        Ok((0..ny).step_by(chunk_y).flat_map(move |y0| {
            (0..nx).step_by(chunk_x).map(move |x0| {
                let y1 = (y0 + chunk_y).min(ny);
                let x1 = (x0 + chunk_x).min(nx);
                CubeChunk {
                    y0,
                    x0,
                    data: data.slice(s![.., y0..y1, x0..x1, ..]),
                }
            })
        }))
    }

    fn check_band(&self, band: usize) -> Result<(), CubeError> {
        let nbands = self.bands.len();
        if band >= nbands {
            return Err(CubeError::BandOutOfRange {
                index: band,
                nbands,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    fn ramp_cube(ny: usize, nx: usize, nt: usize) -> Cube {
        // value = t * (1 + y + x) so every pixel has a distinct linear trend
        let mut data = Array4::zeros((1, ny, nx, nt));
        for ((_, y, x, t), v) in data.indexed_iter_mut() {
            *v = t as f64 * (1.0 + y as f64 + x as f64);
        }
        Cube::new(data, (0..nt).map(|t| t as f64).collect(), vec!["b1".into()]).unwrap()
    }

    #[test]
    fn new_rejects_mismatched_axes() {
        let data = Array4::<f64>::zeros((1, 2, 2, 5));
        assert!(matches!(
            Cube::new(data.clone(), vec![0.0; 4], vec!["b1".into()]),
            Err(CubeError::DimensionMismatch(_))
        ));
        assert!(matches!(
            Cube::new(data, vec![0.0; 5], vec![]),
            Err(CubeError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn series_is_contiguous_time() {
        let cube = ramp_cube(2, 3, 4);
        let s = cube.series(0, 1, 2).unwrap();
        let expected: Vec<f64> = (0..4).map(|t| t as f64 * 4.0).collect();
        assert_eq!(s.as_slice().unwrap(), expected.as_slice());
    }

    #[test]
    fn iter_series_streams_all_pixels() {
        let cube = ramp_cube(3, 4, 2);
        let pixels: Vec<(usize, usize)> =
            cube.iter_series(0).unwrap().map(|p| (p.y, p.x)).collect();
        assert_eq!(pixels.len(), 12);
        assert_eq!(pixels[0], (0, 0));
        assert_eq!(pixels[11], (2, 3));
    }

    #[test]
    fn par_map_series_grid_matches_pixels() {
        let cube = ramp_cube(2, 2, 6);
        let sums = cube
            .par_map_series(0, |_t, y| y.iter().sum::<f64>())
            .unwrap();
        // sum_t t = 15, scaled by (1 + y + x)
        assert_eq!(sums[[0, 0]], 15.0);
        assert_eq!(sums[[1, 1]], 45.0);
    }

    #[test]
    fn chunks_tile_the_grid() {
        let cube = ramp_cube(5, 4, 2);
        let chunks: Vec<_> = cube.chunks(2, 3).unwrap().collect();
        assert_eq!(chunks.len(), 6); // ceil(5/2) * ceil(4/3)
        let total: usize = chunks.iter().map(|c| c.data.dim().1 * c.data.dim().2).sum();
        assert_eq!(total, 20);
        let last = chunks.last().unwrap();
        assert_eq!((last.y0, last.x0), (4, 3));
        assert_eq!(last.data.dim(), (1, 1, 1, 2));
    }

    #[test]
    fn band_out_of_range() {
        let cube = ramp_cube(1, 1, 3);
        assert!(matches!(
            cube.series(2, 0, 0),
            Err(CubeError::BandOutOfRange {
                index: 2,
                nbands: 1
            })
        ));
    }
}
