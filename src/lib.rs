#[cfg(feature = "python")]
pub mod python_api;

pub struct VectorGrid2D {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub nx: usize,
    pub ny: usize,
    pub bx: Vec<f64>,
    pub by: Vec<f64>,
}

impl VectorGrid2D {
    /// Creates a new `VectorGrid2D`.
    /// Panics if `nx` or `ny` < 2, or if `bx`/`by` length does not equal `nx * ny`.
    ///
    /// # Example
    /// ```
    /// use rustronomy::VectorGrid2D;
    ///
    /// let bx = vec![1.0, 1.0, 1.0, 1.0];
    /// let by = vec![0.0, 0.0, 0.0, 0.0];
    /// let grid = VectorGrid2D::new(0.0, 1.0, 0.0, 1.0, 2, 2, bx, by);
    /// ```
    pub fn new(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        nx: usize,
        ny: usize,
        bx: Vec<f64>,
        by: Vec<f64>,
    ) -> Self {
        assert!(nx >= 2, "nx must be at least 2");
        assert!(ny >= 2, "ny must be at least 2");
        assert_eq!(bx.len(), nx * ny, "bx length must be nx * ny");
        assert_eq!(by.len(), nx * ny, "by length must be nx * ny");

        Self {
            x_min,
            x_max,
            y_min,
            y_max,
            nx,
            ny,
            bx,
            by,
        }
    }

    /// Bilinear interpolation of the field at (x, y).
    /// Returns None if the point is outside the grid bounds.
    pub fn interpolate(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        // We use a small epsilon to account for floating point inaccuracies
        // when checking boundaries.
        if x < self.x_min || x > self.x_max || y < self.y_min || y > self.y_max {
            return None;
        }

        let hx = (self.x_max - self.x_min) / (self.nx - 1) as f64;
        let hy = (self.y_max - self.y_min) / (self.ny - 1) as f64;

        let fx = (x - self.x_min) / hx;
        let fy = (y - self.y_min) / hy;

        let mut i = fx.floor() as usize;
        let mut j = fy.floor() as usize;

        // Clamp to prevent out-of-bounds array access if point is exactly on the upper boundary
        if i >= self.nx - 1 {
            i = self.nx - 2;
        }
        if j >= self.ny - 1 {
            j = self.ny - 2;
        }

        let tx = fx - i as f64;
        let ty = fy - j as f64;

        let idx = |i: usize, j: usize| j * self.nx + i;

        let b00x = self.bx[idx(i, j)];
        let b10x = self.bx[idx(i + 1, j)];
        let b01x = self.bx[idx(i, j + 1)];
        let b11x = self.bx[idx(i + 1, j + 1)];

        let bx_top = b00x * (1.0 - tx) + b10x * tx;
        let bx_bot = b01x * (1.0 - tx) + b11x * tx;
        let bx_interp = bx_top * (1.0 - ty) + bx_bot * ty;

        let b00y = self.by[idx(i, j)];
        let b10y = self.by[idx(i + 1, j)];
        let b01y = self.by[idx(i, j + 1)];
        let b11y = self.by[idx(i + 1, j + 1)];

        let by_top = b00y * (1.0 - tx) + b10y * tx;
        let by_bot = b01y * (1.0 - tx) + b11y * tx;
        let by_interp = by_top * (1.0 - ty) + by_bot * ty;

        Some((bx_interp, by_interp))
    }

    /// Traces a field line using a fixed-step 4th-order Runge-Kutta integrator.
    /// Stops and returns the points gathered so far if the line exits the domain.
    pub fn trace_line(&self, seed: (f64, f64), step: f64, max_steps: usize) -> Vec<(f64, f64)> {
        let mut path = Vec::with_capacity(max_steps + 1);

        if self.interpolate(seed.0, seed.1).is_none() {
            return path;
        }

        path.push(seed);
        let mut current = seed;

        for _ in 0..max_steps {
            let (x, y) = current;

            let (k1x, k1y) = match self.interpolate(x, y) {
                Some(v) => v,
                None => break,
            };

            let (k2x, k2y) = match self.interpolate(x + step * 0.5 * k1x, y + step * 0.5 * k1y) {
                Some(v) => v,
                None => break,
            };

            let (k3x, k3y) = match self.interpolate(x + step * 0.5 * k2x, y + step * 0.5 * k2y) {
                Some(v) => v,
                None => break,
            };

            let (k4x, k4y) = match self.interpolate(x + step * k3x, y + step * k3y) {
                Some(v) => v,
                None => break,
            };

            let next_x = x + (step / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            let next_y = y + (step / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);

            // Check if the resulting step is within bounds
            if self.interpolate(next_x, next_y).is_none() {
                break;
            }

            current = (next_x, next_y);
            path.push(current);
        }

        path
    }

}

