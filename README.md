# rustronomy

Simple Rust-backed Python module for tracing field lines.

This package is built with `maturin` and published to PyPI as `rustronomy`.

## Build and install locally

```bash
uv pip install -e .
```

This installs:
- `rustronomy` (Rust extension module)

If you prefer using `maturin` directly:

```bash
maturin develop
```

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

## Publish to PyPI

1. Create an API token on PyPI.
2. Export it in your shell:

```bash
export MATURIN_PYPI_TOKEN=pypi-...
```

3. Build the distribution artifacts locally:

```bash
uvx maturin build
```

4. Optionally validate the built wheel in a clean environment:

```bash
uv venv /tmp/rustronomy-test
source /tmp/rustronomy-test/bin/activate
uv pip install dist/*.whl
python -c "import rustronomy; print(sorted(name for name in dir(rustronomy) if 'fieldlines' in name))"
```

5. Publish to TestPyPI first:

```bash
uvx maturin publish --repository testpypi
```

6. Publish to PyPI:

```bash
uvx maturin publish
```

If the name `rustronomy` is already taken on PyPI, change the `name` field in `pyproject.toml` before publishing.

## Minimal repository layout

- `src/` core Rust tracing implementation and Python bindings
- `Cargo.toml` Rust package configuration
- `pyproject.toml` maturin/Python packaging configuration

## Minimal API

- `trace_fieldlines(...)` fixed-step tracer (RK4)
