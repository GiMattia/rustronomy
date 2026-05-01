# rustronomy

Simple Rust-backed Python module for tracing field lines.

## Build and install locally

```bash
maturin develop
```

This installs:
- `rustronomy` (Rust extension module)

## Minimal API usage

```python
import numpy as np
import rustronomy

paths = rustronomy.trace_fieldlines(
	xmin=0.0,
	xmax=1.0,
	ymin=0.0,
	ymax=1.0,
	nx=16,
	ny=16,
	bx=np.ones(16 * 16),
	by=np.zeros(16 * 16),
	seeds=[(0.1, 0.1), (0.2, 0.3)],
	step=0.01,
	max_steps=1024,
)
```

This project now keeps a small, non-public API focused on one task: compute field lines.

## Minimal repository layout

- `src/` core Rust tracing implementation and Python bindings
- `Cargo.toml` Rust package configuration
- `pyproject.toml` maturin/Python packaging configuration

## Minimal API

- `trace_fieldlines(...)` fixed-step tracer (RK4)
