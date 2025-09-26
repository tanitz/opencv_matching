"""
Reusable template matching utilities for OpenCV + ctypes matcher.

This module packages the core functionality from py_demo.py into a clean
API that other scripts can import without side effects.

Quick usage
-----------
from matching import (
    MatchingParams,
    MatchResult,
    Matcher,
    find_library_path,
    create_matcher_for_template,
    run_match,
    draw_results,
)

dll_path = find_library_path()  # or provide absolute path
params = MatchingParams(maxCount=5, scoreThreshold=0.6, iouThreshold=0.4, angle=10.0)
matcher = create_matcher_for_template(template_img, dll_path, params)
count, results = run_match(matcher, source_img)
vis = draw_results(source_img, results)

Notes
-----
- This module does not perform any long-running operations at import time.
- Library loading is done when constructing a Matcher.
"""

from __future__ import annotations

import os
import sys
import platform
import ctypes
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2

__all__ = [
    "MatchingParams",
    "MatchResult",
    "Matcher",
    "find_library_path",
    "create_matcher_for_template",
    "run_match",
    "draw_results",
    "result_to_points",
    "release_matcher",
    "load_image",
    "load_image_gray",
]


# ------------------------- Environment helpers -------------------------
def sanitize_ld_library_path(remove_keywords: Optional[List[str]] = None) -> None:
    """Remove problematic entries (e.g., conda/anaconda) from LD_LIBRARY_PATH
    for this process so that system libs used by OpenCV/GDAL are resolved consistently.
    No-op on Windows.
    """
    if platform.system() == "Windows":
        return
    if remove_keywords is None:
        remove_keywords = ["anaconda", "conda"]
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld:
        return
    parts = [p for p in ld.split(":") if p]
    kept: List[str] = []
    for p in parts:
        lower = p.lower()
        if any(k in lower for k in remove_keywords):
            continue
        kept.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(kept)


def preload_system_sqlite_if_available() -> None:
    """Preload system libsqlite3 with RTLD_GLOBAL to satisfy other deps.
    No-op on Windows. Safe to call multiple times.
    """
    if platform.system() == "Windows":
        return
    candidates = [
        "/lib/x86_64-linux-gnu/libsqlite3.so.0",
        "/usr/lib/x86_64-linux-gnu/libsqlite3.so.0",
        "/lib64/libsqlite3.so.0",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                ctypes.CDLL(c, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                return
            except OSError:
                pass


def prepare_environment() -> None:
    """Optional: minimal environment hygiene to reduce loader conflicts.
    This is safe to call before constructing a Matcher, especially in
    heterogeneous environments mixing system and conda libs.
    """
    sanitize_ld_library_path()
    preload_system_sqlite_if_available()


# --------------------------- Library discovery ---------------------------
default_name_windows = "templatematching_ctype.dll"
default_name_linux = "libtemplatematching_ctype.so"
default_name_linux_alt = "templatematching_ctype.so"


def find_library_path(provided_path: Optional[str] = None, *, cwd: Optional[str] = None) -> str:
    """Find the path to the templatematching ctypes library.

    - If provided_path is absolute and exists, use it.
    - If provided_path is relative and exists under cwd or current working dir, use it.
    - Else search common build/install locations.
    - Else return the platform-default name to allow system loader to resolve by name.
    """
    if cwd is None:
        cwd = os.getcwd()

    if provided_path:
        if os.path.isabs(provided_path) and os.path.exists(provided_path):
            return provided_path
        rel = os.path.join(cwd, provided_path)
        if os.path.exists(rel):
            return rel

    system = platform.system()
    candidates: List[str] = []
    if system == "Windows":
        candidates = [
            default_name_windows,
            os.path.join("build", "matcher", "templatematching_ctype.dll"),
        ]
    else:
        candidates = [
            default_name_linux,
            default_name_linux_alt,
            os.path.join("build", "matcher", "libtemplatematching_ctype.so"),
            os.path.join("install", "lib", "libtemplatematching_ctype.so"),
            os.path.join("matcher", "libtemplatematching_ctype.so"),
            os.path.join("build", "matcher", "libtemplatematching.so"),
            os.path.join("install", "matcher", "libtemplatematching_ctype.so"),
        ]

    search_paths = [
        cwd,
        os.path.join(cwd, "build", "matcher"),
        os.path.join(cwd, "matcher"),
        os.path.join(cwd, "install", "matcher"),
        os.path.join(cwd, "install", "lib"),
    ]

    for name in candidates:
        if os.path.isabs(name):
            if os.path.exists(name):
                return name
        else:
            p = os.path.join(cwd, name)
            if os.path.exists(p):
                return p
            for sp in search_paths:
                p2 = os.path.join(sp, name)
                if os.path.exists(p2):
                    return p2

    return default_name_windows if system == "Windows" else default_name_linux


def _preload_dependency_if_available(dep_path: Optional[str] = None) -> None:
    """Preload libtemplatematching.so with RTLD_GLOBAL on POSIX systems.
    Safe to call when not present.
    """
    if platform.system() == "Windows":
        return
    candidates = [
        dep_path,
        os.path.join(os.getcwd(), "build", "matcher", "libtemplatematching.so"),
        os.path.join(os.getcwd(), "install", "lib", "libtemplatematching.so"),
        os.path.join(os.getcwd(), "matcher", "libtemplatematching.so"),
        os.path.join(os.getcwd(), "build", "libtemplatematching.so"),
        "libtemplatematching.so",
    ]
    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            try:
                dep_dir = os.path.dirname(os.path.abspath(p))
                ld_env = os.environ.get("LD_LIBRARY_PATH", "")
                if dep_dir and dep_dir not in ld_env.split(":"):
                    os.environ["LD_LIBRARY_PATH"] = f"{dep_dir}:{ld_env}" if ld_env else dep_dir
                ctypes.CDLL(p, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                return
            except OSError:
                # Not fatal; higher-level loader may still succeed if system path contains it
                pass


# ------------------------------ Data types ------------------------------
class MatchResult(ctypes.Structure):
    _fields_ = [
        ("leftTopX", ctypes.c_double),
        ("leftTopY", ctypes.c_double),
        ("leftBottomX", ctypes.c_double),
        ("leftBottomY", ctypes.c_double),
        ("rightTopX", ctypes.c_double),
        ("rightTopY", ctypes.c_double),
        ("rightBottomX", ctypes.c_double),
        ("rightBottomY", ctypes.c_double),
        ("centerX", ctypes.c_double),
        ("centerY", ctypes.c_double),
        ("angle", ctypes.c_double),
        ("score", ctypes.c_double),
    ]


@dataclass
class MatchingParams:
    maxCount: int = 1
    scoreThreshold: float = 0.5
    iouThreshold: float = 0.4
    angle: float = 0.0
    minArea: float = 256.0


# ------------------------------- Matcher -------------------------------
class Matcher:
    """Thin ctypes wrapper around the C++ template matcher library.

    Construct with desired parameters and a path/name for the shared library.
    Then call set_template() followed by match() for each source image.
    """

    def __init__(
        self,
        dll_path: str,
        maxCount: int,
        scoreThreshold: float,
        iouThreshold: float,
        angle: float,
        minArea: float,
        *,
        preload_dependency: bool = True,
        prepare_env: bool = True,
    ) -> None:
        if maxCount <= 0:
            raise ValueError("maxCount must be greater than 0")

        if prepare_env:
            prepare_environment()

        if preload_dependency:
            _preload_dependency_if_available()

        # Resolve dll_path to an absolute file if possible (search common build/install locations
        # relative to this module) and add its directory to LD_LIBRARY_PATH so dependencies load.
        resolved = find_library_path(dll_path, cwd=os.path.dirname(__file__)) if dll_path else find_library_path(None, cwd=os.path.dirname(__file__))
        dll_abspath = None
        if resolved and os.path.exists(resolved):
            dll_abspath = os.path.abspath(resolved)
        else:
            # search subtree for matching filenames (slower fallback)
            here = os.path.abspath(os.path.dirname(__file__))
            target_names = [dll_path, default_name_linux, default_name_linux_alt]
            for root, _, files in os.walk(here):
                for n in target_names:
                    if not n:
                        continue
                    if n in files:
                        dll_abspath = os.path.join(root, n)
                        break
                if dll_abspath:
                    break
        # If still not found, keep original dll_path (let loader try by name)
        if not dll_abspath:
            dll_abspath = dll_path

        # If we found a file, ensure its directory is in LD_LIBRARY_PATH so dependent .so load
        if isinstance(dll_abspath, str) and os.path.exists(dll_abspath):
            dep_dir = os.path.dirname(os.path.abspath(dll_abspath))
            ld_env = os.environ.get("LD_LIBRARY_PATH", "")
            if dep_dir and dep_dir not in ld_env.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{dep_dir}:{ld_env}" if ld_env else dep_dir

        try:
            # Load with RTLD_GLOBAL to make symbols available to subsequently loaded libs
            self.lib = ctypes.CDLL(dll_abspath, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError as e:
            raise FileNotFoundError(
                f"Could not load templatematching library '{dll_path}'. Tried: {dll_abspath}. Error: {e}"
            )

        # Configure C signatures
        self.lib.matcher.argtypes = [
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.matcher.restype = ctypes.c_void_p
        self.lib.setTemplate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.match.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(MatchResult),
            ctypes.c_int,
        ]

        self.maxCount = int(maxCount)
        self.scoreThreshold = float(scoreThreshold)
        self.iouThreshold = float(iouThreshold)
        self.angle = float(angle)
        self.minArea = float(minArea)

        self._matcher = self.lib.matcher(
            self.maxCount, self.scoreThreshold, self.iouThreshold, self.angle, self.minArea
        )
        self.results = (MatchResult * self.maxCount)()

    # -- API --
    def set_template(self, image: np.ndarray) -> int:
        """Set the template from a grayscale image array."""
        if image is None:
            raise ValueError("template image is None")
        if image.ndim == 3:
            # Convert to grayscale if needed
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Invalid template image shape")
        h, w = int(image.shape[0]), int(image.shape[1])
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.setTemplate(self._matcher, data, w, h, channels)

    def match(self, image: np.ndarray) -> int:
        """Run matching against the provided image. Returns number of matches (>=0) or <0 on failure."""
        if image is None:
            raise ValueError("image is None")
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Invalid image shape")
        h, w = int(image.shape[0]), int(image.shape[1])
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.match(self._matcher, data, w, h, channels, self.results, self.maxCount)

    def get_results(self, count: Optional[int] = None) -> List[MatchResult]:
        """Return a list of MatchResult up to 'count' or maxCount, filtered by score>0."""
        if count is None:
            count = self.maxCount
        out: List[MatchResult] = []
        for i in range(min(count, self.maxCount)):
            r = self.results[i]
            if r.score > 0:
                out.append(r)
        return out


# ------------------------------ Convenience ------------------------------
def create_matcher_for_template(
    template_img: Union[str, np.ndarray],
    dll_path: Optional[str] = None,
    params: Optional[MatchingParams] = None,
    *,
    provided_path: Optional[str] = None,
) -> Matcher:
    """Create a Matcher with given template and parameters.

    - If dll_path is None, tries to resolve via find_library_path(provided_path).
    """
    if params is None:
        params = MatchingParams()
    if dll_path is None:
        dll_path = find_library_path(provided_path)

    # Load template if a file path was given
    if isinstance(template_img, str):
        tpl = load_image_gray(template_img)
        if tpl is None:
            raise ValueError(f"Failed to read template image: {template_img}")
    else:
        tpl = template_img

    m = Matcher(
        dll_path,
        params.maxCount,
        params.scoreThreshold,
        params.iouThreshold,
        params.angle,
        params.minArea,
    )
    m.set_template(tpl)
    return m


def run_match(matcher: Matcher, image: Union[str, np.ndarray]) -> Tuple[int, List[MatchResult]]:
    """Run matcher on an image array or image file path and return (count, results list)."""
    if isinstance(image, str):
        img = load_image_gray(image)
        if img is None:
            raise ValueError(f"Failed to read source image: {image}")
    else:
        img = image

    count = matcher.match(img)
    if count < 0:
        return count, []
    return count, matcher.get_results(count)


def result_to_points(result: MatchResult) -> np.ndarray:
    """Convert MatchResult into a 4x2 int32 array of polygon points."""
    coords = np.array([
        result.leftTopX,
        result.leftTopY,
        result.leftBottomX,
        result.leftBottomY,
        result.rightBottomX,
        result.rightBottomY,
        result.rightTopX,
        result.rightTopY,
    ], dtype=np.float64)

    # Ensure all coordinates are finite numbers; downstream code casts to int
    if not np.all(np.isfinite(coords)):
        raise ValueError("MatchResult contains non-finite coordinates")

    pts = coords.reshape(4, 2).astype(np.int32)
    return pts


def get_centers(results: Sequence[MatchResult]) -> List[Tuple[int, int]]:
    """Return a list of integer (x, y) center points for valid results."""
    centers: List[Tuple[int, int]] = []
    for r in results:
        try:
            if not np.isfinite(r.centerX) or not np.isfinite(r.centerY):
                continue
            centers.append((int(r.centerX), int(r.centerY)))
        except Exception:
            continue
    return centers


def draw_results(
    image: Union[str, np.ndarray],
    results: Sequence[MatchResult],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    with_score: bool = True,
    draw_center: bool = True,
) -> np.ndarray:
    """Draw match polygons (and scores) on a copy of the image and return it.

    The `image` can be a numpy array (BGR) or a file path string.
    """
    if isinstance(image, str):
        src = load_image(image)
        if src is None:
            raise ValueError(f"Failed to read image for drawing: {image}")
    else:
        src = image
    out = src.copy()
    for r in results:
        # skip invalid scores or non-finite results
        try:
            if not np.isfinite(r.score) or r.score <= 0:
                continue
            pts = result_to_points(r)
        except Exception:
            # Skip results with invalid coordinates (NaN/Inf) or other issues
            continue
        cv2.polylines(out, [pts], True, color, thickness)
        if with_score:
            cv2.putText(
                out,
                f"{r.score:.3f}",
                (int(r.leftTopX), int(r.leftTopY) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        if draw_center:
            # Try to draw center if available and finite
            try:
                if np.isfinite(r.centerX) and np.isfinite(r.centerY):
                    cx, cy = int(r.centerX), int(r.centerY)
                    cv2.circle(out, (cx, cy), 4, color, -1)
            except Exception:
                pass
    return out


# ------------------------------ Image helpers ------------------------------
def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from a file path in BGR (color) using OpenCV."""
    return cv2.imread(path)


def load_image_gray(path: str) -> Optional[np.ndarray]:
    """Load a grayscale image from a file path using OpenCV."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def release_matcher(matcher: Optional[Matcher]) -> None:
    """Release resources of a Matcher instance.

    The underlying library doesn't expose an explicit destroy function; Python will
    free resources when the object is garbage collected. This function exists as a
    convenience so callers can explicitly drop references.
    """
    # Best-effort: drop references to ctypes handle to encourage GC
    if matcher is None:
        return
    try:
        # Remove large buffers and the ctypes handle
        matcher.results = None  # type: ignore[attr-defined]
        matcher.lib = None      # type: ignore[attr-defined]
    except Exception:
        pass

