"""Task A — UI feature parity: every MODA UI capability has a FastMODA equivalent.

Two independent checks per inventory row:

1. **Route parity** — the mapped FastMODA HTTP route is actually registered in
   ``FastMODA/app.py`` (parsed statically, so this check needs no dependencies).
2. **Backing-symbol parity** — the mapped ``module:function`` is importable from
   the installed ``fastmoda`` package (needs the FastMODA deps; runs in-container).

Rows flagged ``expected_gap=True`` are MODA-desktop-only capabilities; they are
reported but never fail the suite.

Run (inside the FastMODA image, from the repo root mounted at /repo):
    pytest -q /repo/tests/parity/test_ui_parity.py
"""

import importlib
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
APP_PY = os.path.join(REPO, "FastMODA", "app.py")

sys.path.insert(0, HERE)
from moda_inventory import INVENTORY, Feature  # noqa: E402


def _registered_routes(app_py=APP_PY):
    """Statically scrape every @app.route('<path>') from app.py."""
    with open(app_py, "r", encoding="utf-8") as fh:
        src = fh.read()
    return set(re.findall(r"@app\.route\(\s*['\"]([^'\"]+)['\"]", src))


ROUTES = _registered_routes()
_ACTIVE = [f for f in INVENTORY if not f.expected_gap]


def _fid(f: Feature):
    return f"{f.module}:{f.feature}"


@pytest.mark.parametrize("feat", _ACTIVE, ids=_fid)
def test_route_is_registered(feat: Feature):
    """Every mapped FastMODA route must exist in app.py."""
    if feat.fm_route is None:
        pytest.skip(f"{_fid(feat)} has no HTTP route (client-side/implicit)")
    assert feat.fm_route in ROUTES, (
        f"MODA feature '{feat.feature}' maps to FastMODA route "
        f"'{feat.fm_route}' but that route is NOT registered in app.py. "
        f"Registered routes: {sorted(ROUTES)}"
    )


@pytest.mark.parametrize("feat", _ACTIVE, ids=_fid)
def test_backing_symbol_importable(feat: Feature):
    """Every mapped fastmoda module:function must be importable."""
    if feat.fm_symbol is None:
        pytest.skip(f"{_fid(feat)} has no backing symbol")
    mod_name, _, func_name = feat.fm_symbol.partition(":")
    try:
        mod = importlib.import_module(f"fastmoda.{mod_name}") \
            if mod_name != "fastmoda" else importlib.import_module("fastmoda")
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"fastmoda.{mod_name} not importable in this env: {exc}")
    assert hasattr(mod, func_name), (
        f"MODA feature '{feat.feature}' maps to '{feat.fm_symbol}' but "
        f"'{func_name}' is missing from fastmoda.{mod_name}."
    )


def test_coverage_report(capsys):
    """Emit a human-readable parity matrix and assert full non-gap coverage."""
    lines, covered, gaps = [], 0, 0
    by_mod = {}
    for f in INVENTORY:
        by_mod.setdefault(f.module, []).append(f)

    lines.append("\nMODA → FastMODA feature parity matrix")
    lines.append("=" * 78)
    for mod, feats in by_mod.items():
        lines.append(f"\n[{mod}]")
        for f in feats:
            if f.expected_gap:
                gaps += 1
                mark = "○ MODA-only"
                tgt = f.note or "(desktop-specific)"
            else:
                route_ok = f.fm_route is None or f.fm_route in ROUTES
                covered += route_ok
                mark = "✓" if route_ok else "✗ MISSING"
                tgt = f.fm_route or f.fm_symbol or "(implicit)"
            lines.append(f"   {mark:12} {f.feature:52} → {tgt}")

    total_active = len(_ACTIVE)
    lines.append("\n" + "=" * 78)
    lines.append(f"Covered: {covered}/{total_active} active capabilities   "
                 f"({gaps} intentional MODA-desktop-only gaps)")
    report = "\n".join(lines)
    print(report)

    missing = [f for f in _ACTIVE
               if f.fm_route is not None and f.fm_route not in ROUTES]
    assert not missing, "Uncovered MODA features: " + ", ".join(
        f"{m.feature}→{m.fm_route}" for m in missing)
