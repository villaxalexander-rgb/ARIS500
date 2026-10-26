"""Load the optional C++ portfolio weight solver.

Compiles ``native/aris_core.cpp`` into ``native/build/`` on first use when a
C++ compiler is available and calls it through ctypes. Writes nothing else.
When the library cannot be built or loaded, or ``ARIS500_NATIVE=0`` is set,
``available()`` returns False and portfolio construction uses the Python
reference implementation, which produces identical weights.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = _ROOT / "native" / "aris_core.cpp"
BUILD_DIR = _ROOT / "native" / "build"
_SUFFIX = ".dylib" if sys.platform == "darwin" else ".dll" if os.name == "nt" else ".so"
LIBRARY_PATH = BUILD_DIR / f"libaris_core{_SUFFIX}"

# Bitwise parity with numpy needs strict IEEE arithmetic: no fused multiply-add
# contraction and no fast-math reassociation.
COMPILE_FLAGS = ["-O3", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off"]

OK = 0
LOWER_ABOVE_UPPER = 1
INFEASIBLE_TOTAL = 2
BETA_INFEASIBLE = 3

_double_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_int_array = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_library: ctypes.CDLL | None = None
_load_attempted = False
load_error: str | None = None


def _compiler() -> str | None:
    """Return the first available C++ compiler command."""
    for name in (os.environ.get("CXX"), "c++", "clang++", "g++"):
        if name and shutil.which(name):
            return name
    return None


def build(force: bool = False) -> Path:
    """Compile the native solver if it is missing or older than its source.

    Args:
        force: Rebuild even when the existing library is current.

    Returns:
        Computed result as ``Path``.
    """
    if (
        not force
        and LIBRARY_PATH.exists()
        and LIBRARY_PATH.stat().st_mtime >= SOURCE_PATH.stat().st_mtime
    ):
        return LIBRARY_PATH
    compiler = _compiler()
    if compiler is None:
        raise RuntimeError("No C++ compiler found; set CXX or install clang/g++.")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [compiler, *COMPILE_FLAGS, str(SOURCE_PATH), "-o", str(LIBRARY_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    return LIBRARY_PATH


def _load() -> ctypes.CDLL | None:
    """Build and bind the library once, recording why it is unavailable."""
    global _library, _load_attempted, load_error
    if _load_attempted:
        return _library
    _load_attempted = True
    if os.environ.get("ARIS500_NATIVE", "1") == "0":
        load_error = "disabled by ARIS500_NATIVE=0"
        return None
    try:
        library = ctypes.CDLL(str(build()))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        load_error = f"native solver unavailable: {detail.strip()}"
        return None
    library.aris_bounded_scale.restype = ctypes.c_int
    library.aris_bounded_scale.argtypes = [
        _double_array, _double_array, _double_array,
        ctypes.c_int64, ctypes.c_double, _double_array,
    ]
    library.aris_name_sector_caps.restype = ctypes.c_int
    library.aris_name_sector_caps.argtypes = [
        _double_array, _int_array, ctypes.c_int64,
        _double_array, ctypes.c_int32,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        _double_array, _double_array,
    ]
    library.aris_beta_constrained_weights.restype = ctypes.c_int
    library.aris_beta_constrained_weights.argtypes = [
        _double_array, _double_array, _int_array, ctypes.c_int64,
        _double_array, ctypes.c_int32, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        _double_array, _double_array,
    ]
    _library = library
    return _library


def available() -> bool:
    """Return whether the native solver is built, loaded, and enabled.

    Args:
        None.

    Returns:
        Computed result as ``bool``.
    """
    return _load() is not None


def _doubles(values) -> np.ndarray:
    """Return a contiguous float64 copy-free view where possible."""
    return np.ascontiguousarray(values, dtype=np.float64)


def bounded_scale(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float,
) -> tuple[int, np.ndarray]:
    """Scale positive targets to a total within per-element bounds.

    Args:
        target: Positive target weights.
        lower: Per-element lower bounds.
        upper: Per-element upper bounds.
        total: Required sum of the result.

    Returns:
        Computed result as ``tuple[int, np.ndarray]`` (status, weights).
    """
    target = _doubles(target)
    out = np.empty_like(target)
    status = _load().aris_bounded_scale(
        target, _doubles(lower), _doubles(upper), len(target), float(total), out
    )
    return status, out


def name_sector_caps(
    raw_target: np.ndarray,
    sector_codes: np.ndarray,
    benchmark_sector: np.ndarray,
    max_name_weight: float,
    min_position_weight: float,
    max_sector_deviation: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Project raw weights onto full-investment, position, and sector bounds.

    Args:
        raw_target: Unnormalized raw target weight per selected name.
        sector_codes: Zero-based sector position per selected name.
        benchmark_sector: Benchmark weight per sector position.
        max_name_weight: Maximum allowed security weight.
        min_position_weight: Minimum allowed security weight.
        max_sector_deviation: Allowed absolute sector deviation.

    Returns:
        Computed result as ``tuple[int, np.ndarray, np.ndarray]`` (status,
        name weights, portfolio sector weights).
    """
    raw_target = _doubles(raw_target)
    benchmark_sector = _doubles(benchmark_sector)
    weights = np.zeros_like(raw_target)
    portfolio_sector = np.zeros_like(benchmark_sector)
    status = _load().aris_name_sector_caps(
        raw_target,
        np.ascontiguousarray(sector_codes, dtype=np.int32),
        len(raw_target),
        benchmark_sector,
        len(benchmark_sector),
        float(max_name_weight),
        float(min_position_weight),
        float(max_sector_deviation),
        weights,
        portfolio_sector,
    )
    return status, weights, portfolio_sector


def beta_constrained_weights(
    initial_weights: np.ndarray,
    betas: np.ndarray,
    sector_codes: np.ndarray,
    benchmark_sector: np.ndarray,
    target_beta: float,
    max_name_weight: float,
    min_position_weight: float,
    max_sector_deviation: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Tilt weights along beta until the target bound is reached.

    Args:
        initial_weights: Capped weights before the beta tilt.
        betas: Constraint beta per selected name, with no missing values.
        sector_codes: Zero-based sector position per selected name.
        benchmark_sector: Benchmark weight per sector position.
        target_beta: Beta bound to reach.
        max_name_weight: Maximum allowed security weight.
        min_position_weight: Minimum allowed security weight.
        max_sector_deviation: Allowed absolute sector deviation.

    Returns:
        Computed result as ``tuple[int, np.ndarray, np.ndarray]`` (status,
        name weights, portfolio sector weights).
    """
    initial_weights = _doubles(initial_weights)
    benchmark_sector = _doubles(benchmark_sector)
    weights = np.zeros_like(initial_weights)
    portfolio_sector = np.zeros_like(benchmark_sector)
    status = _load().aris_beta_constrained_weights(
        initial_weights,
        _doubles(betas),
        np.ascontiguousarray(sector_codes, dtype=np.int32),
        len(initial_weights),
        benchmark_sector,
        len(benchmark_sector),
        float(target_beta),
        float(max_name_weight),
        float(min_position_weight),
        float(max_sector_deviation),
        weights,
        portfolio_sector,
    )
    return status, weights, portfolio_sector
