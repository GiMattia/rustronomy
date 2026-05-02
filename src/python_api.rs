use crate::VectorGrid2D;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule, PyString};
use pyo3::wrap_pyfunction;

/// Trace field lines on a 2D vector grid.
///
/// Parameters:
/// - `py`: Python interpreter token (provided automatically).
/// - `xmin`, `xmax`, `ymin`, `ymax`: domain bounds.
/// - `nx`, `ny`: number of grid points in each direction.
/// - `bx`, `by`: flattened Bx and By components of shape (nx * ny).
/// - `seeds`: list of (x, y) seed points.
/// - `step`: integration step size.
/// - `max_steps`: maximum number of RK4 steps per seed.
///
/// Returns a 3‑D NumPy array of shape (num_seeds, max_steps + 1, 2) containing the
/// traced points. Unused entries are filled with NaN.
#[pyfunction]
fn trace_fieldlines(
    py: Python,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: usize,
    ny: usize,
    bx: PyReadonlyArray1<f64>,
    by: PyReadonlyArray1<f64>,
    seeds: Vec<(f64, f64)>,
    step: f64,
    max_steps: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    // Convert NumPy read‑only arrays to Rust vectors
    let bx_vec = bx.as_slice()?.to_vec();
    let by_vec = by.as_slice()?.to_vec();

    let grid = VectorGrid2D::new(xmin, xmax, ymin, ymax, nx, ny, bx_vec, by_vec);
    let num_seeds = seeds.len();

    // Allocate output array filled with NaNs
    let mut out = numpy::ndarray::Array3::<f64>::from_elem((num_seeds, max_steps + 1, 2), f64::NAN);
    for (i, seed) in seeds.into_iter().enumerate() {
        let path = grid.trace_line(seed, step, max_steps);
        for (j, (x, y)) in path.into_iter().enumerate() {
            out[[i, j, 0]] = x;
            out[[i, j, 1]] = y;
        }
    }
    // Convert to a Python‑owned NumPy array
    Ok(out.into_pyarray(py).to_owned().into())
}

fn extract_vector(arg: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if arg.hasattr("ravel")? {
        // Rust indexing uses idx = j * nx + i, which corresponds to Fortran-order
        // flattening for arrays shaped as (nx, ny).
        let flattened = arg.call_method1("ravel", ("F",))?;
        let list = flattened.call_method0("tolist")?;
        return list.extract::<Vec<f64>>();
    }

    if let Ok(array) = arg.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(array.as_slice()?.to_vec());
    }

    if let Ok(array) = arg.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        return Ok(array.as_array().iter().copied().collect());
    }

    if let Ok(values) = arg.extract::<Vec<f64>>() {
        return Ok(values);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a 1D sequence or NumPy array of numeric values",
    ))
}

fn extract_field(data: &Bound<'_, PyAny>, field: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(name) = field.cast::<PyString>() {
        let attr = data.getattr(name.to_str()?)?;
        return extract_vector(&attr);
    }

    extract_vector(field)
}

fn trace_line_until_close(
    grid: &VectorGrid2D,
    seed: (f64, f64),
    step: f64,
    max_steps: usize,
    close_on_loop: bool,
    close_tol: f64,
    min_time_before_close: f64,
) -> (Vec<(f64, f64)>, bool) {
    let mut path = Vec::with_capacity(max_steps + 2);

    if grid.interpolate(seed.0, seed.1).is_none() {
        return (path, false);
    }

    path.push(seed);
    let mut current = seed;
    let mut closed = false;

    for step_idx in 0..max_steps {
        let (x, y) = current;

        let (k1x, k1y) = match grid.interpolate(x, y) {
            Some(v) => v,
            None => break,
        };

        let (k2x, k2y) = match grid.interpolate(x + step * 0.5 * k1x, y + step * 0.5 * k1y) {
            Some(v) => v,
            None => break,
        };

        let (k3x, k3y) = match grid.interpolate(x + step * 0.5 * k2x, y + step * 0.5 * k2y) {
            Some(v) => v,
            None => break,
        };

        let (k4x, k4y) = match grid.interpolate(x + step * k3x, y + step * k3y) {
            Some(v) => v,
            None => break,
        };

        let next_x = x + (step / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        let next_y = y + (step / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);

        if grid.interpolate(next_x, next_y).is_none() {
            break;
        }

        current = (next_x, next_y);
        path.push(current);

        if close_on_loop {
            let dx = current.0 - seed.0;
            let dy = current.1 - seed.1;
            let elapsed_time = (step_idx as f64 + 1.0) * step.abs();
            if elapsed_time > min_time_before_close && (dx * dx + dy * dy).sqrt() <= close_tol {
                path.push(seed);
                closed = true;
                break;
            }
        }
    }

    (path, closed)
}

#[pyfunction(signature = (
    data,
    var1,
    var2,
    *,
    x1 = None,
    x2 = None,
    x0 = None,
    y0 = None,
    maxstep = None,
    numsteps = None,
    step = None,
    rtol = None,
    atol = None,
    order = None,
    dense = None,
    close = None,
    ctol = None,
    text = false,
    transpose = false
))]
fn find_fieldlines(
    py: Python,
    data: &Bound<'_, PyAny>,
    var1: &Bound<'_, PyAny>,
    var2: &Bound<'_, PyAny>,
    x1: Option<&Bound<'_, PyAny>>,
    x2: Option<&Bound<'_, PyAny>>,
    x0: Option<&Bound<'_, PyAny>>,
    y0: Option<&Bound<'_, PyAny>>,
    maxstep: Option<f64>,
    numsteps: Option<usize>,
    step: Option<f64>,
    rtol: Option<f64>,
    atol: Option<f64>,
    order: Option<&str>,
    dense: Option<bool>,
    close: Option<bool>,
    ctol: Option<f64>,
    text: bool,
    transpose: bool,
) -> PyResult<Py<PyAny>> {
    let _ = (rtol, atol, order, dense, text, transpose);

    let xc = match x1 {
        Some(values) => extract_vector(values)?,
        None => extract_vector(&data.getattr("x1")?)?,
    };
    let yc = match x2 {
        Some(values) => extract_vector(values)?,
        None => extract_vector(&data.getattr("x2")?)?,
    };

    if xc.len() < 2 || yc.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x1 and x2 must each contain at least 2 points",
        ));
    }

    let x0 = match x0 {
        Some(values) => extract_vector(values)?,
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Footpoints not provided. Please provide x0 and y0.",
            ));
        }
    };
    let y0 = match y0 {
        Some(values) => extract_vector(values)?,
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Footpoints not provided. Please provide x0 and y0.",
            ));
        }
    };

    if x0.len() != y0.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x0 and y0 must have the same length",
        ));
    }

    let bx = extract_field(data, var1)?;
    let by = extract_field(data, var2)?;

    let nx = xc.len();
    let ny = yc.len();

    if bx.len() != nx * ny || by.len() != nx * ny {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Field arrays must have length nx * ny (got bx={}, by={}, nx*ny={})",
            bx.len(),
            by.len(),
            nx * ny
        )));
    }

    let dx = (xc[1] - xc[0]).abs();
    let dy = (yc[1] - yc[0]).abs();
    let step = step.unwrap_or(dx.min(dy));
    let max_steps = numsteps.unwrap_or(16384);
    let max_step = maxstep.unwrap_or(100.0 * step);
    let signed_step = step.min(max_step);
    let close_on_loop = close.unwrap_or(true);
    let close_tol = ctol.unwrap_or(1.0e-6);
    let min_time_before_close = max_step;

    let grid = VectorGrid2D::new(
        xc[0],
        xc[nx - 1],
        yc[0],
        yc[ny - 1],
        nx,
        ny,
        bx,
        by,
    );

    let lines = PyList::empty(py);
    for (&seed_x, &seed_y) in x0.iter().zip(y0.iter()) {
        let (forward, closed_forward) = trace_line_until_close(
            &grid,
            (seed_x, seed_y),
            signed_step,
            max_steps,
            close_on_loop,
            close_tol,
            min_time_before_close,
        );

        if closed_forward && forward.len() > 2 {
            let mut x_line = Vec::with_capacity(forward.len());
            let mut y_line = Vec::with_capacity(forward.len());
            for &(x, y) in &forward {
                x_line.push(x);
                y_line.push(y);
            }

            let pair = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, x_line).into_any(),
                    PyArray1::from_vec(py, y_line).into_any(),
                ],
            )?;
            lines.append(pair)?;
            continue;
        }

        let (backward, _) = trace_line_until_close(
            &grid,
            (seed_x, seed_y),
            -signed_step,
            max_steps,
            false,
            close_tol,
            min_time_before_close,
        );

        let mut x_line = Vec::with_capacity(backward.len() + forward.len().saturating_sub(1));
        let mut y_line = Vec::with_capacity(backward.len() + forward.len().saturating_sub(1));

        for &(x, y) in backward.iter().rev() {
            x_line.push(x);
            y_line.push(y);
        }

        for &(x, y) in forward.iter().skip(1) {
            x_line.push(x);
            y_line.push(y);
        }

        if x_line.len() > 1 {
            let pair = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, x_line).into_any(),
                    PyArray1::from_vec(py, y_line).into_any(),
                ],
            )?;
            lines.append(pair)?;
        }
    }

    Ok(lines.into_any().unbind())
}

/// Python module definition.
#[pymodule]
fn rustronomy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trace_fieldlines, m)?)?;
    m.add_function(wrap_pyfunction!(find_fieldlines, m)?)?;
    Ok(())
}
