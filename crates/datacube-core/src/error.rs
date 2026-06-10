use thiserror::Error;

/// Errors produced by cube construction, access and statistics.
#[derive(Debug, Error)]
pub enum CubeError {
    /// Shapes of data, time axis or band labels do not agree.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// A band index outside `0..nbands`.
    #[error("band index {index} out of range ({nbands} bands)")]
    BandOutOfRange { index: usize, nbands: usize },

    /// Not enough finite observations for the requested statistic.
    #[error("insufficient data: need at least {needed} finite observations, got {got}")]
    InsufficientData { needed: usize, got: usize },

    /// Chunk sizes must be strictly positive.
    #[error("invalid chunk size: {0}")]
    InvalidChunkSize(String),
}
