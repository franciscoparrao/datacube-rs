//! Per-cell band algebra: spectral indices and general band combinations.
//!
//! These transforms combine the bands of a cube at each `(y, x, time)` cell
//! and return a new single-band cube on the same spatial/temporal grid. They
//! are NaN-aware: a cell is `NaN` whenever any input band is `NaN` (a masked
//! observation) or the index is undefined (e.g. a zero normalized-difference
//! denominator). The named indices in [`indices`] are thin wrappers over
//! [`Cube::normalized_difference`] and [`Cube::combine_bands`].

use ndarray::{Array4, s};
use rayon::prelude::*;

use crate::cube::Cube;
use crate::error::CubeError;

impl Cube {
    /// Index of the band labelled `name`.
    pub fn band(&self, name: &str) -> Result<usize, CubeError> {
        self.bands()
            .iter()
            .position(|b| b == name)
            .ok_or_else(|| CubeError::BandNotFound(name.to_string()))
    }

    /// Normalized difference `(a - b) / (a + b)` of two bands, as a new
    /// single-band cube labelled `label`.
    ///
    /// A cell is `NaN` if either input is `NaN` or the denominator is zero.
    /// This is the workhorse behind NDVI, NDWI, NBR and NDBI (see [`indices`]).
    ///
    /// ```
    /// # use datacube_core::Cube;
    /// # use ndarray::Array4;
    /// // 2 bands (red, nir), 1 pixel, 1 time
    /// let mut data = Array4::zeros((2, 1, 1, 1));
    /// data[[0, 0, 0, 0]] = 0.2; // red
    /// data[[1, 0, 0, 0]] = 0.6; // nir
    /// let cube = Cube::new(data, vec![0.0], vec!["red".into(), "nir".into()]).unwrap();
    /// let ndvi = cube.normalized_difference(1, 0, "ndvi").unwrap();
    /// assert!((ndvi.data()[[0, 0, 0, 0]] - 0.5).abs() < 1e-12); // (0.6-0.2)/(0.6+0.2)
    /// ```
    pub fn normalized_difference(
        &self,
        a: usize,
        b: usize,
        label: &str,
    ) -> Result<Cube, CubeError> {
        self.binary_band(a, b, label, |x, y| {
            let denom = x + y;
            if denom == 0.0 {
                f64::NAN
            } else {
                (x - y) / denom
            }
        })
    }

    /// Simple ratio `a / b` of two bands, as a new single-band cube.
    ///
    /// A cell is `NaN` if either input is `NaN` or the denominator is zero.
    pub fn band_ratio(&self, a: usize, b: usize, label: &str) -> Result<Cube, CubeError> {
        self.binary_band(a, b, label, |x, y| if y == 0.0 { f64::NAN } else { x / y })
    }

    /// Applies a per-band linear transform `v' = slope·v + offset` to the named
    /// bands (bands not listed are copied unchanged), returning a new cube on
    /// the same grid. `NaN` (masked observations) is preserved.
    ///
    /// This is the cross-sensor **harmonization** operator: to fuse a
    /// multi-decadal Landsat + Sentinel-2 record into one consistent per-pixel
    /// trajectory, the bandpass difference between sensors must be removed
    /// before the two are concatenated in time. Harmonize the single-sensor
    /// cube of one sensor (e.g. Sentinel-2 MSI) into the other's spectral space
    /// (e.g. Landsat OLI) band by band, then concatenate. The engine supplies
    /// the *mechanism*; the coefficients are cited external data — published
    /// OLS bandpass-adjustment slopes/offsets (e.g. Claverie et al. 2018, HLS
    /// Table 5; Roy et al. 2016) — passed in per band, so the transform is
    /// reproducible and auditable rather than hard-coded.
    ///
    /// Errors if a listed band name is not present. Coefficients must be finite.
    pub fn harmonize(&self, coeffs: &[(String, f64, f64)]) -> Result<Cube, CubeError> {
        // resolve band names up front (fail before mutating anything)
        let resolved: Vec<(usize, f64, f64)> = coeffs
            .iter()
            .map(|(name, slope, offset)| {
                if !slope.is_finite() || !offset.is_finite() {
                    return Err(CubeError::InvalidParameter(format!(
                        "harmonize coefficients for band '{name}' must be finite \
                         (slope={slope}, offset={offset})"
                    )));
                }
                Ok((self.band(name)?, *slope, *offset))
            })
            .collect::<Result<_, _>>()?;

        let mut data = self.data().to_owned();
        for (bi, slope, offset) in resolved {
            // band is the outermost axis, so each band volume is contiguous;
            // a NaN stays NaN under the affine map.
            data.slice_mut(s![bi, .., .., ..])
                .mapv_inplace(|v| slope * v + offset);
        }
        Ok(Cube::new(data, self.time().to_vec(), self.bands().to_vec())?.inherit_georef(self))
    }

    /// Combines all bands at every `(y, x, time)` cell with `f`, producing a
    /// new single-band cube labelled `label`.
    ///
    /// `f` receives the band values of one cell in band order. This is the
    /// general primitive for indices that mix more than two bands or use
    /// constants (EVI, SAVI); `f` is responsible for its own NaN handling
    /// (returning `NaN` propagates a masked or undefined cell).
    ///
    /// ```
    /// # use datacube_core::Cube;
    /// # use ndarray::Array4;
    /// let mut data = Array4::zeros((2, 1, 1, 1));
    /// data[[0, 0, 0, 0]] = 3.0;
    /// data[[1, 0, 0, 0]] = 4.0;
    /// let cube = Cube::new(data, vec![0.0], vec!["a".into(), "b".into()]).unwrap();
    /// let sum = cube.combine_bands("sum", |v| v[0] + v[1]).unwrap();
    /// assert_eq!(sum.data()[[0, 0, 0, 0]], 7.0);
    /// ```
    pub fn combine_bands<F>(&self, label: &str, f: F) -> Result<Cube, CubeError>
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        let (nb, ny, nx, nt) = self.dims();
        let cells = ny * nx * nt;
        let view = self.data();
        let src = view
            .as_slice()
            .expect("cube data is standard layout (enforced by Cube::new)");

        // For cell i the band values are at i, i + cells, i + 2*cells, ...
        // (band is the outermost axis). Parallelize over output cells.
        let mut out = vec![0.0f64; cells];
        out.par_iter_mut().enumerate().for_each(|(i, dst)| {
            let mut vals = [0.0f64; MAX_STACK_BANDS];
            if nb <= MAX_STACK_BANDS {
                for (band, slot) in vals.iter_mut().take(nb).enumerate() {
                    *slot = src[band * cells + i];
                }
                *dst = f(&vals[..nb]);
            } else {
                let cell: Vec<f64> = (0..nb).map(|band| src[band * cells + i]).collect();
                *dst = f(&cell);
            }
        });

        let data = Array4::from_shape_vec((1, ny, nx, nt), out)
            .map_err(|e| CubeError::DimensionMismatch(e.to_string()))?;
        Ok(Cube::new(data, self.time().to_vec(), vec![label.to_string()])?.inherit_georef(self))
    }

    /// Element-wise binary op over two whole band volumes (each contiguous in
    /// standard layout), with NaN propagated automatically.
    fn binary_band<F>(&self, a: usize, b: usize, label: &str, f: F) -> Result<Cube, CubeError>
    where
        F: Fn(f64, f64) -> f64 + Sync,
    {
        let (nb, ny, nx, nt) = self.dims();
        if a >= nb {
            return Err(CubeError::BandOutOfRange {
                index: a,
                nbands: nb,
            });
        }
        if b >= nb {
            return Err(CubeError::BandOutOfRange {
                index: b,
                nbands: nb,
            });
        }
        let cells = ny * nx * nt;
        let view = self.data();
        let src = view
            .as_slice()
            .expect("cube data is standard layout (enforced by Cube::new)");
        let va = &src[a * cells..(a + 1) * cells];
        let vb = &src[b * cells..(b + 1) * cells];

        let mut out = vec![0.0f64; cells];
        out.par_iter_mut().enumerate().for_each(|(i, dst)| {
            let (x, y) = (va[i], vb[i]);
            *dst = if x.is_nan() || y.is_nan() {
                f64::NAN
            } else {
                f(x, y)
            };
        });

        let data = Array4::from_shape_vec((1, ny, nx, nt), out)
            .map_err(|e| CubeError::DimensionMismatch(e.to_string()))?;
        Ok(Cube::new(data, self.time().to_vec(), vec![label.to_string()])?.inherit_georef(self))
    }
}

/// Cells with at most this many bands gather into a stack buffer (no alloc);
/// wider cubes fall back to a heap vector per cell. Real ARD cubes have a
/// handful of bands, so the fast path is the common one.
const MAX_STACK_BANDS: usize = 16;

/// Named spectral indices, resolved by band label.
///
/// Each function looks the named bands up in the cube and returns a new
/// single-band cube. Normalized-difference indices delegate to
/// [`Cube::normalized_difference`]; EVI and SAVI use [`Cube::combine_bands`].
pub mod indices {
    use super::*;

    /// NDVI = (NIR − Red) / (NIR + Red). Vegetation greenness.
    pub fn ndvi(cube: &Cube, nir: &str, red: &str) -> Result<Cube, CubeError> {
        cube.normalized_difference(cube.band(nir)?, cube.band(red)?, "ndvi")
    }

    /// NDWI = (Green − NIR) / (Green + NIR) (McFeeters). Open water.
    pub fn ndwi(cube: &Cube, green: &str, nir: &str) -> Result<Cube, CubeError> {
        cube.normalized_difference(cube.band(green)?, cube.band(nir)?, "ndwi")
    }

    /// NBR = (NIR − SWIR) / (NIR + SWIR). Burn severity.
    pub fn nbr(cube: &Cube, nir: &str, swir: &str) -> Result<Cube, CubeError> {
        cube.normalized_difference(cube.band(nir)?, cube.band(swir)?, "nbr")
    }

    /// NDBI = (SWIR − NIR) / (SWIR + NIR). Built-up area.
    pub fn ndbi(cube: &Cube, swir: &str, nir: &str) -> Result<Cube, CubeError> {
        cube.normalized_difference(cube.band(swir)?, cube.band(nir)?, "ndbi")
    }

    /// EVI = 2.5·(NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1). Canopy index
    /// less sensitive to soil/atmosphere; expects surface reflectance.
    pub fn evi(cube: &Cube, nir: &str, red: &str, blue: &str) -> Result<Cube, CubeError> {
        let (n, r, b) = (cube.band(nir)?, cube.band(red)?, cube.band(blue)?);
        cube.combine_bands("evi", move |v| {
            let denom = v[n] + 6.0 * v[r] - 7.5 * v[b] + 1.0;
            if denom == 0.0 {
                f64::NAN
            } else {
                2.5 * (v[n] - v[r]) / denom
            }
        })
    }

    /// SAVI = (1 + L)·(NIR − Red) / (NIR + Red + L). Soil-adjusted vegetation
    /// index; `l` is the soil-brightness factor (0.5 is the usual default).
    pub fn savi(cube: &Cube, nir: &str, red: &str, l: f64) -> Result<Cube, CubeError> {
        let (n, r) = (cube.band(nir)?, cube.band(red)?);
        cube.combine_bands("savi", move |v| {
            let denom = v[n] + v[r] + l;
            if denom == 0.0 {
                f64::NAN
            } else {
                (1.0 + l) * (v[n] - v[r]) / denom
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::GeoRef;
    use ndarray::Array4;

    /// red, nir, blue cube: 1 pixel, 2 time steps; second step is masked NIR.
    fn rgbn_cube() -> Cube {
        let mut data = Array4::zeros((3, 1, 1, 2));
        // t=0
        data[[0, 0, 0, 0]] = 0.1; // red
        data[[1, 0, 0, 0]] = 0.5; // nir
        data[[2, 0, 0, 0]] = 0.05; // blue
        // t=1 (nir masked)
        data[[0, 0, 0, 1]] = 0.2;
        data[[1, 0, 0, 1]] = f64::NAN;
        data[[2, 0, 0, 1]] = 0.05;
        Cube::new(
            data,
            vec![0.0, 1.0],
            vec!["red".into(), "nir".into(), "blue".into()],
        )
        .unwrap()
    }

    #[test]
    fn band_lookup() {
        let c = rgbn_cube();
        assert_eq!(c.band("nir").unwrap(), 1);
        assert!(matches!(c.band("swir"), Err(CubeError::BandNotFound(_))));
    }

    #[test]
    fn harmonize_applies_per_band_and_preserves_nan_and_others() {
        let c = rgbn_cube();
        // adjust only red: red' = 0.98*red - 0.01 ; nir/blue unchanged
        let h = c.harmonize(&[("red".into(), 0.98, -0.01)]).unwrap();
        assert_eq!(h.dims(), c.dims());
        assert!((h.data()[[0, 0, 0, 0]] - (0.98 * 0.1 - 0.01)).abs() < 1e-12);
        assert!((h.data()[[0, 0, 0, 1]] - (0.98 * 0.2 - 0.01)).abs() < 1e-12);
        // nir unchanged (and its NaN preserved)
        assert_eq!(h.data()[[1, 0, 0, 0]], 0.5);
        assert!(h.data()[[1, 0, 0, 1]].is_nan());
        // blue unchanged
        assert_eq!(h.data()[[2, 0, 0, 0]], 0.05);
    }

    #[test]
    fn harmonize_removes_cross_sensor_step() {
        // "true" reflectance identical at both dates; sensor B (t=1) records it
        // through a bandpass difference v_B = a*v_true + b. Harmonizing band by
        // the inverse (1/a, -b/a) must recover v_true → no step between dates.
        let (a, b) = (0.95_f64, 0.02_f64);
        let v_true = 0.4_f64;
        let mut data = Array4::zeros((1, 1, 1, 2));
        data[[0, 0, 0, 0]] = v_true; // sensor A (already OLI space)
        data[[0, 0, 0, 1]] = a * v_true + b; // sensor B (MSI space)
        let cube = Cube::new(data, vec![0.0, 1.0], vec!["green".into()]).unwrap();
        // NB: in the real pipeline B is a separate single-sensor sub-cube; here
        // both dates share the band, so harmonize would touch both — instead we
        // verify the inverse recovers v_true on a B-only cube.
        let mut b_only = Array4::zeros((1, 1, 1, 1));
        b_only[[0, 0, 0, 0]] = a * v_true + b;
        let b_cube = Cube::new(b_only, vec![1.0], vec!["green".into()]).unwrap();
        let harmonized = b_cube
            .harmonize(&[("green".into(), 1.0 / a, -b / a)])
            .unwrap();
        assert!((harmonized.data()[[0, 0, 0, 0]] - v_true).abs() < 1e-12);
        let _ = cube; // documents the two-sensor setup
    }

    #[test]
    fn harmonize_rejects_unknown_band_and_nonfinite() {
        let c = rgbn_cube();
        assert!(matches!(
            c.harmonize(&[("swir".into(), 1.0, 0.0)]),
            Err(CubeError::BandNotFound(_))
        ));
        assert!(matches!(
            c.harmonize(&[("red".into(), f64::NAN, 0.0)]),
            Err(CubeError::InvalidParameter(_))
        ));
    }

    #[test]
    fn harmonize_preserves_georef() {
        let geo = GeoRef {
            epsg: Some(32719),
            transform: Some([300_000.0, 10.0, 0.0, 6_200_000.0, 0.0, -10.0]),
        };
        let c = rgbn_cube().with_georef(geo);
        let h = c.harmonize(&[("nir".into(), 1.0, 0.0)]).unwrap();
        assert_eq!(h.georef(), Some(geo));
    }

    #[test]
    fn ndvi_values_and_nan_propagation() {
        let c = rgbn_cube();
        let nd = indices::ndvi(&c, "nir", "red").unwrap();
        assert_eq!(nd.dims(), (1, 1, 1, 2));
        assert_eq!(nd.bands(), ["ndvi"]);
        // (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6
        assert!((nd.data()[[0, 0, 0, 0]] - (0.4 / 0.6)).abs() < 1e-12);
        // masked NIR -> NaN
        assert!(nd.data()[[0, 0, 0, 1]].is_nan());
    }

    #[test]
    fn normalized_difference_zero_denominator_is_nan() {
        let mut data = Array4::zeros((2, 1, 1, 1));
        data[[0, 0, 0, 0]] = 0.3;
        data[[1, 0, 0, 0]] = -0.3; // a + b = 0
        let c = Cube::new(data, vec![0.0], vec!["a".into(), "b".into()]).unwrap();
        let nd = c.normalized_difference(0, 1, "nd").unwrap();
        assert!(nd.data()[[0, 0, 0, 0]].is_nan());
    }

    #[test]
    fn evi_matches_closed_form() {
        let c = rgbn_cube();
        let evi = indices::evi(&c, "nir", "red", "blue").unwrap();
        // 2.5*(0.5-0.1)/(0.5 + 6*0.1 - 7.5*0.05 + 1) = 2.5*0.4/1.725
        let expected = 2.5 * 0.4 / (0.5 + 0.6 - 0.375 + 1.0);
        assert!((evi.data()[[0, 0, 0, 0]] - expected).abs() < 1e-12);
        assert!(evi.data()[[0, 0, 0, 1]].is_nan()); // masked nir
    }

    #[test]
    fn savi_reduces_to_ndvi_when_l_zero() {
        let c = rgbn_cube();
        let savi = indices::savi(&c, "nir", "red", 0.0).unwrap();
        let ndvi = indices::ndvi(&c, "nir", "red").unwrap();
        assert!((savi.data()[[0, 0, 0, 0]] - ndvi.data()[[0, 0, 0, 0]]).abs() < 1e-12);
    }

    #[test]
    fn band_out_of_range_is_reported() {
        let c = rgbn_cube();
        assert!(matches!(
            c.normalized_difference(9, 0, "x"),
            Err(CubeError::BandOutOfRange {
                index: 9,
                nbands: 3
            })
        ));
    }

    #[test]
    fn indices_propagate_georef() {
        let geo = GeoRef {
            epsg: Some(32719),
            transform: Some([300_000.0, 10.0, 0.0, 6_200_000.0, 0.0, -10.0]),
        };
        let c = rgbn_cube().with_georef(geo);
        assert_eq!(indices::ndvi(&c, "nir", "red").unwrap().georef(), Some(geo));
        assert_eq!(
            indices::evi(&c, "nir", "red", "blue").unwrap().georef(),
            Some(geo)
        );
        assert_eq!(c.band_ratio(0, 1, "r").unwrap().georef(), Some(geo));
    }
}
