# heavy_assembly_project

A hero-scale Saturn V F-1 engine: eight imported sub-assemblies flattened in
`main.kcl`. The engine executes it but **cannot currently measure it as a
whole**, which makes it the fixture for the physical-properties failure path.

Exercised by `tests/test_live_heavy_assembly.py`.

## Provenance

Reported as [KittyCAD/text-to-cad#3928][issue] by @JordanNoone, where physical
analysis failed after both mock and real execution had succeeded. These files
are the project attached to that issue (ZIP SHA-256
`4f07d025dde0f6b95a5ecb7a33d2c95a37a96a24291db8cafbd44b2900cad864`), extracted
verbatim.

[issue]: https://github.com/KittyCAD/text-to-cad/issues/3928

## Measured on 2026-07-30

Live engine, `zoo-kcl` 0.3.170 (this branch's `uv.lock`) / `zoo-mcp` 0.18.2 /
`kittycad` 1.4.0 (the issue was filed against 0.3.158 / 0.15.4 / 1.3.8):

| Scope | Attempts | Result |
| --- | --- | --- |
| Whole project (`main.kcl`) | 4 | 4 failures |
| `injectorHead.kcl` alone | 1 | success in ~25s |

Whole-project attempts:

1. 30.0s — `engine: ... message: "internal error: unknown"` at `SourceRange([0, 0, 0])`
2. 24.4s — same
3. 619.6s — ``Modeling command timed out `07e5d8fe-a080-4f40-89d0-b8cd4bc15227` (API call ID: 4908aaca-78b9-4acf-818e-1fcb07b0926f)``
4. 24.3s — `internal error: unknown`

Two findings that correct the issue as filed:

- **It is not intermittent.** Every whole-project attempt failed, so the issue's
  "verified workaround: retry the call" does not hold.
- **Scope reduction is the working recovery.** `injectorHead.kcl` returned real
  properties (volume 1573102.86 mm³, mass 12348.86 kg, surface area
  22037682.64 mm², bbox 2064 x 2745 x 1010 mm) — the engine can measure the
  parts but not the assembled whole.

`kcl.KclError.is_retryable()` is `False` for both messages, so
`_execute_with_retries` never retried them.

## Where the defect lives

In the engine's measure path, below `zoo-mcp`. The failure carries no source
location and, in the fast case, no identifiers, so nothing above the engine can
attribute it. `zoo_calculate_kcl_physical_properties` now raises
`ZooKclEngineError`, which lifts the engine's message, retryable flag, and any
modeling command / API call IDs into fields — but reporting the failure well is
not fixing it. The API call ID above is the handle for that investigation.
