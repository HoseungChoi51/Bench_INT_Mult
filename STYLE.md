# STYLE — coding, writing, and visualization conventions

These are the **non-negotiable defaults** for every notebook and library module added to this repo, now and as it expands. `PLAN.md` is a design artifact; this file is enforced.

---

## A. Audience & voice

- **Reader profile:** SW developer fluent in Python and integer arithmetic, not assumed to know FHE, RNS, hardware multipliers, or low-level perf engineering.
- **Voice:** declarative; second person used sparingly ("we compute…", "notice that…"). No marketing tone, no exclamation points, no rhetorical questions in section headers.
- **Define on first use.** Every acronym (RNS, NTT, CKKS, GEMM, FHE, …) gets a one-sentence inline gloss the first time it appears in a notebook, even if defined in an earlier notebook. Notebooks must be readable standalone.
- **No claims without a number or a citation.** Performance/throughput statements include a concrete factor or a `# source:` comment.
- **Conclusion first, derivation second.** Each section opens with a one-line takeaway, then walks the reasoning. Each notebook ends with a "Takeaway" cell of 3–6 bullet points.

---

## B. Notebook structure

Every notebook follows the same skeleton:

1. **Title cell** — `# Notebook N — <title>`, one-paragraph abstract, prerequisites (which earlier notebooks), estimated reading time.
2. **Imports cell** — exactly one, near the top. No re-imports later.
3. **Three layers per concept**, in order:
   1. concept explanation in prose + math,
   2. toy implementation (importing from `rns_arithmetic`, never redefining library functions inline),
   3. cost model / practical implication.
4. **Interactive widget cell** clearly labeled `### Interactive: …`.
5. **Takeaway cell** at the end.

Cell-level rules:

- Markdown cells **explain or summarize**; code cells **compute or visualize**. Do not narrate inside code via long comment paragraphs — put prose in a markdown cell above.
- Math uses LaTeX in `$…$` / `$$…$$`. Do not write math as ASCII (`a^2 + b`) when a math block is appropriate.
- No cell may exceed ~40 lines. Refactor longer logic into the library.
- Cells are **idempotent and order-independent within a section**: re-running a cell must not corrupt state. Avoid mutating module globals from a notebook.

---

## C. Code style (library + notebook code cells)

- **Python 3.13.** Every library module starts with `from __future__ import annotations`.
- **Type hints required** on every public function and dataclass field. `int`, `float`, `list[int]`, `np.ndarray`, etc. — no bare `Any` unless justified in a comment.
- **Docstrings:** NumPy style. Required on every public function in `src/rns_arithmetic/`. Include `Parameters`, `Returns`, and (where useful) `Examples` sections. Include the underlying formula in LaTeX where relevant.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes/dataclasses, `UPPER_SNAKE` for module-level constants. Prefer descriptive names over abbreviations (`prime_bits`, not `pb`). Mathematical single-letter names (`a`, `b`, `q`, `w`) are allowed inside short numerical functions where they match the math.
- **Pure functions by default.** No hidden global state, no module-level mutation. Dataclasses are `frozen=True` unless mutation is essential.
- **No `print` for results.** Library code returns values; notebooks display via the last-expression-in-cell idiom or `IPython.display`.
- **Imports:** stdlib → third-party → local, separated by blank lines. Always `import numpy as np`, `import matplotlib.pyplot as plt`, `import pandas as pd`. No star imports.
- **Errors over silence.** Validate inputs at boundaries with explicit `raise ValueError(…)`. Do not silently coerce types.
- **`ruff` enforces it.** Configured in `pyproject.toml` with `line-length = 100` and rule sets `E, F, I, UP, B, SIM, NPY`. CI gate: `uv run ruff check` clean.

---

## D. Mathematical & numerical conventions

- **Bit widths spelled out.** `prime_bits`, `chunk_bits`, `operand_bits`, `product_bits` — never just `width` or `n` without scope.
- **Endianness convention.** Limb / chunk lists are **least-significant first**: `words[0]` is the LSB. Document this once here and reference it; do not re-explain in every docstring.
- **NumPy dtypes are explicit.** Always pass `dtype=np.uint64` etc.; do not rely on inference for arithmetic-correctness-critical arrays. Add an assertion when overflow correctness depends on dtype.
- **Reproducibility.** Any cell using randomness creates a local `rng = np.random.default_rng(seed)` with a documented seed. No reliance on global RNG state.

---

## E. Visualization style

- **One concept per figure.** No combined dual-axis plots unless the comparison is the point.
- **Standard size:** `figsize=(8, 5)` for line plots, `(6, 5)` for square comparisons (heatmaps, grids). Use `constrained_layout=True`.
- **Color palette:** a single project palette defined once in `visualization.py` (`PALETTE`), imported everywhere. Default to a **colorblind-safe** qualitative set (Okabe–Ito); for sequential data use `viridis`. Never use rainbow / `jet`.
- **Semantic color assignments are stable across notebooks.** Defined once in `visualization.SEMANTIC_COLORS`. Current bindings:

  | Semantic key   | Meaning                              |
  | -------------- | ------------------------------------ |
  | `naive`        | Naive shift-add multiplication       |
  | `booth_r2`     | Radix-2 Booth                        |
  | `booth_r4`     | Radix-4 Booth                        |
  | `prime_31`     | 31-bit RNS prime strategy            |
  | `prime_36`     | 36-bit RNS prime strategy            |
  | `int8_chunk`   | INT8 chunk decomposition             |
  | `reference`    | Reference / boundary lines           |

- **Always label:** title, x-axis (with units), y-axis (with units), legend with descriptive entries (no `line0`). Add the headline number as an annotation when there is one (e.g. the 1.16× threshold line).
- **Reference lines** (e.g. the 64-bit boundary in product-width plots) are dashed gray with an inline label.
- **Tables** (Markdown or pandas) follow the same column order across notebooks for like quantities: `prime_bits | product_bits | fits_uint64 | limbs | cost`.
- **Widgets:** use `ipywidgets` only; widget cells must produce a static fallback rendering (the figure for the default slider values) so committed outputs are meaningful on GitHub even without a kernel.

---

## F. Testing & verification

- Every library module has a `tests/test_<module>.py`.
- Property tests for arithmetic correctness use `np.random.default_rng(seed)` (or `random.Random(seed)`) with a documented seed and ≥1000 iterations.
- Notebooks are smoke-tested by `jupyter nbconvert --execute --inplace`; the gate is "all four notebooks execute without raising".

---

## G. Docs & filenames

- Notebook filenames: `NN_snake_case_topic.ipynb`, zero-padded `NN`.
- Library module filenames: `snake_case.py`. One topic per module.
- New top-level concept folders go under `notebooks/` with the same `NN_` numbering scheme; library code lives under `src/rns_arithmetic/<topic>/` if it warrants a subpackage.
- This file (`STYLE.md`) is the source of truth. When adding a new convention, update this file in the same commit.
