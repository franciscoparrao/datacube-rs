use thiserror::Error;

/// Errors produced by cube construction, access and statistics.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CubeError {
    /// Shapes of data, time axis or band labels do not agree.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// An operation that requires a sorted time axis got an unsorted one.
    #[error("time axis must be ascending: {0}")]
    UnsortedTime(String),

    /// The input is structurally valid but carries no usable signal
    /// (e.g. a constant time coordinate, where a slope is undefined).
    #[error("degenerate input: {0}")]
    DegenerateInput(String),

    /// A band index outside `0..nbands`.
    #[error("band index {index} out of range ({nbands} bands)")]
    BandOutOfRange { index: usize, nbands: usize },

    /// No band carries the requested label.
    #[error("band not found: {0}")]
    BandNotFound(String),

    /// Not enough finite observations for the requested statistic.
    #[error("insufficient data: need at least {needed} finite observations, got {got}")]
    InsufficientData { needed: usize, got: usize },

    /// Chunk sizes must be strictly positive.
    #[error("invalid chunk size: {0}")]
    InvalidChunkSize(String),

    /// A statistical parameter outside its valid domain.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// The regression design matrix is rank-deficient (e.g. aliased
    /// harmonics or constant time).
    #[error("singular system: {0}")]
    SingularSystem(String),
}
