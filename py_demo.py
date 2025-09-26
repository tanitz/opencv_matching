import sys
import os
import platform
import ctypes
import numpy as np
import cv2
from typing import List

# --- Runtime loader hygiene to avoid conda/system lib conflicts ---
def _sanitize_ld_library_path(remove_keywords: List[str] = None) -> None:
    """
    Remove problematic entries (e.g., conda/anaconda) from LD_LIBRARY_PATH for this process
    so that system libs used by OpenCV/GDAL are resolved consistently.
    """
    if platform.system() == 'Windows':
        return
    if remove_keywords is None:
        remove_keywords = ['anaconda', 'conda']
    ld = os.environ.get('LD_LIBRARY_PATH', '')
    if not ld:
        return
    parts = [p for p in ld.split(':') if p]
    kept = []
    removed = []
    for p in parts:
        lower = p.lower()
        if any(k in lower for k in remove_keywords):
            removed.append(p)
        else:
            kept.append(p)
    if removed:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(kept)
        print(f"Sanitized LD_LIBRARY_PATH: removed {len(removed)} paths (conda/anaconda).")
        # Note: we only affect future dlopen() calls (ctypes.CDLL). Already-loaded libs remain.

def _preload_system_sqlite_if_available():
    """
    Preload system libsqlite3 with RTLD_GLOBAL to satisfy GDAL dependencies when conda provides an older sqlite.
    """
    if platform.system() == 'Windows':
        return
    candidates = [
        '/lib/x86_64-linux-gnu/libsqlite3.so.0',
        '/usr/lib/x86_64-linux-gnu/libsqlite3.so.0',
        '/lib64/libsqlite3.so.0',
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                ctypes.CDLL(c, mode=getattr(ctypes, 'RTLD_GLOBAL', 0))
                print(f"Preloaded system SQLite: {c}")
                return
            except OSError as e:
                print(f"Warning: failed to preload {c}: {e}")
    # Not fatal if not found; continue.

# Perform hygiene early, before any ctypes loads
_sanitize_ld_library_path()
_preload_system_sqlite_if_available()

# 定义MatchResult结构体
class MatchResult(ctypes.Structure):
    _fields_ = [
        ('leftTopX', ctypes.c_double),
        ('leftTopY', ctypes.c_double),
        ('leftBottomX', ctypes.c_double),
        ('leftBottomY', ctypes.c_double),
        ('rightTopX', ctypes.c_double),
        ('rightTopY', ctypes.c_double),
        ('rightBottomX', ctypes.c_double),
        ('rightBottomY', ctypes.c_double),
        ('centerX', ctypes.c_double),
        ('centerY', ctypes.c_double),
        ('angle', ctypes.c_double),
        ('score', ctypes.c_double)
    ]

# 定义Matcher类
class Matcher:
    def __init__(self, dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea):
        # Load with RTLD_GLOBAL to make symbols available to subsequently loaded libs
        self.lib = ctypes.CDLL(dll_path, mode=getattr(ctypes, 'RTLD_GLOBAL', 0))
        self.lib.matcher.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.lib.matcher.restype = ctypes.c_void_p
        self.lib.setTemplate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.match.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(MatchResult), ctypes.c_int]
        
        if maxCount <= 0:
            raise ValueError("maxCount must be greater than 0")
        self.maxCount = maxCount
        self.scoreThreshold = scoreThreshold
        self.iouThreshold = iouThreshold
        self.angle = angle
        self.minArea = minArea
        
        self.matcher = self.lib.matcher(maxCount, scoreThreshold, iouThreshold, angle, minArea)

        self.results = (MatchResult * self.maxCount)()
    
    def set_template(self, image):
        height, width = image.shape[0], image.shape[1]
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.setTemplate(self.matcher, data, width, height, channels)
    
    def match(self, image):
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Invalid image shape")
        height, width = image.shape[0], image.shape[1]
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.match(self.matcher, data, width, height, channels, self.results, self.maxCount)

# 示例调用（允许通过环境变量覆盖）
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

maxCount = _env_int('TM_MAXCOUNT', 1)
scoreThreshold = _env_float('TM_SCORE', 0.5)   # เปอร์เซ็นต์ความมั่นใจ (0.0 - 1.0)
iouThreshold = _env_float('TM_IOU', 0.4)
angle = _env_float('TM_ANGLE', 0.0)            # ช่วงองศาการหมุน (±angle องศา)
minArea = _env_float('TM_MINAREA', 256)

# Try to find the shared library automatically depending on platform.
# On Linux the build produces .so files (see /build/matcher), on Windows .dll.
default_name_windows = 'templatematching_ctype.dll'
default_name_linux = 'libtemplatematching_ctype.so'
default_name_linux_alt = 'templatematching_ctype.so'

def find_library_path(provided_path=None):
    # If user provided explicit path, try that first
    if provided_path:
        if os.path.isabs(provided_path) and os.path.exists(provided_path):
            return provided_path
        rel = os.path.join(os.getcwd(), provided_path)
        if os.path.exists(rel):
            return rel

    system = platform.system()
    candidates = []
    if system == 'Windows':
        candidates = [default_name_windows, os.path.join('build', 'matcher', 'templatematching_ctype.dll')]
    else:
        # Linux / macOS
        candidates = [default_name_linux, default_name_linux_alt,
                      os.path.join('build', 'matcher', 'libtemplatematching_ctype.so'),
                      os.path.join('install', 'lib', 'libtemplatematching_ctype.so'),
                      os.path.join('matcher', 'libtemplatematching_ctype.so'),
                      os.path.join('build', 'matcher', 'libtemplatematching.so'),
                      os.path.join('install', 'matcher', 'libtemplatematching_ctype.so')]

    # Also check the current directory and workspace root
    search_paths = [os.getcwd(), os.path.join(os.getcwd(), 'build', 'matcher'), os.path.join(os.getcwd(), 'matcher'), os.path.join(os.getcwd(), 'install', 'matcher'), os.path.join(os.getcwd(), 'install', 'lib')]

    for name in candidates:
        # absolute or relative
        if os.path.isabs(name):
            if os.path.exists(name):
                return name
        else:
            # try in CWD
            p = os.path.join(os.getcwd(), name)
            if os.path.exists(p):
                return p
            # try search paths
            for sp in search_paths:
                p2 = os.path.join(sp, name)
                if os.path.exists(p2):
                    return p2

    # lastly, try loader defaults (let CDLL try to resolve by name)
    # return the platform-specific filename if nothing else found
    return default_name_windows if system == 'Windows' else default_name_linux


# 模板匹配库路径 (可以是相对或绝对路径). If you have a specific path, put it here.
# Try to find and load the library
dll_path = find_library_path(os.environ.get('TM_LIB_PATH'))
print(f"Using library path candidate: {dll_path}")

# On Linux/macOS, proactively preload the dependency libtemplatematching.so so symbols are globally available
if platform.system() != 'Windows':
    dep_candidates = [
        os.path.join(os.getcwd(), 'build', 'matcher', 'libtemplatematching.so'),
        os.path.join(os.getcwd(), 'install', 'lib', 'libtemplatematching.so'),
        os.path.join(os.getcwd(), 'matcher', 'libtemplatematching.so'),
        os.path.join(os.getcwd(), 'build', 'libtemplatematching.so'),
        'libtemplatematching.so'
    ]
    dep_path = next((p for p in dep_candidates if os.path.exists(p)), None)
    if dep_path:
        try:
            # Ensure the directory is included in LD_LIBRARY_PATH for this process (helps dlopen by name)
            dep_dir = os.path.dirname(os.path.abspath(dep_path))
            ld_env = os.environ.get('LD_LIBRARY_PATH', '')
            ld_paths = [p for p in ld_env.split(':') if p]
            if dep_dir not in ld_paths:
                os.environ['LD_LIBRARY_PATH'] = f"{dep_dir}:{ld_env}" if ld_env else dep_dir
                print(f"Added to LD_LIBRARY_PATH for this process: {dep_dir}")
            ctypes.CDLL(dep_path, mode=getattr(ctypes, 'RTLD_GLOBAL', 0))
            print(f"Preloaded dependency: {dep_path}")
        except OSError as e:
            print(f"Warning: Failed to preload dependency {dep_path}: {e}")
            print("If you see errors about libtemplatematching.so later, set LD_LIBRARY_PATH to include its directory.")
    else:
        # Not fatal yet; the subsequent error handler will try again if needed
        print("Note: Could not find libtemplatematching.so to preload. Will attempt lazy load; may require LD_LIBRARY_PATH.")
try:
    matcher = Matcher(dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea)
except OSError as e:
    # If loading by explicit path failed, try letting the system loader find it by name
    print(f"Failed to load {dll_path}: {e}")
    alt_name = default_name_linux if platform.system() != 'Windows' else default_name_windows
    print(f"Trying loader default name: {alt_name}")
    try:
        matcher = Matcher(alt_name, maxCount, scoreThreshold, iouThreshold, angle, minArea)
    except OSError as e2:
        print(f"Failed to load library by name {alt_name}: {e2}")
        # Common cause: the ctype .so depends on libtemplatematching.so which isn't in loader path.
        missing_dep = 'libtemplatematching.so'
        err_text = str(e2)
        if missing_dep in err_text or 'cannot open shared object file' in err_text:
            # Try to find the dependency next to candidate locations and preload it with RTLD_GLOBAL
            dep_candidates = [
                os.path.join(os.getcwd(), 'build', 'matcher', 'libtemplatematching.so'),
                os.path.join(os.getcwd(), 'install', 'lib', 'libtemplatematching.so'),
                os.path.join(os.getcwd(), 'matcher', 'libtemplatematching.so'),
                os.path.join(os.getcwd(), 'build', 'matcher', 'libtemplatematching.so'),
                os.path.join(os.getcwd(), 'install', 'matcher', 'libtemplatematching.so'),
                os.path.join(os.getcwd(), 'build', 'libtemplatematching.so'),
                'libtemplatematching.so'
            ]
            dep_path = None
            for dc in dep_candidates:
                if os.path.exists(dc):
                    dep_path = dc
                    break

            if dep_path:
                print(f"Preloading dependency: {dep_path}")
                try:
                    # Load dependency globally so the ctype .so can resolve symbols
                    ctypes.CDLL(dep_path, mode=getattr(ctypes, 'RTLD_GLOBAL', 0))
                    # Retry loading the ctype library
                    print(f"Retrying to load {alt_name} after preloading dependency")
                    matcher = Matcher(alt_name, maxCount, scoreThreshold, iouThreshold, angle, minArea)
                except OSError as e3:
                    print(f"Still failed after preloading dependency: {e3}")
                    print("Please add the directory containing libtemplatematching.so to LD_LIBRARY_PATH or install the library system-wide.")
                    sys.exit(111)
            else:
                print("Could not find libtemplatematching.so in common build/install paths.")
                print("Please build the project (see README) or set TM_LIB_PATH or LD_LIBRARY_PATH so the shared object can be found.")
                sys.exit(111)
        else:
            print("Please build the library or set TM_LIB_PATH or LD_LIBRARY_PATH so the shared object can be found.")
            sys.exit(111)

# For testing: use up to two templates and a source path (file or folder)
default_t1 = '/home/iiot-b20/Documents/Robotvision/flet-camera-app/image_comppressor_picture/temp1.jpg'
default_t2 = '/home/iiot-b20/Documents/Robotvision/flet-camera-app/image_comppressor_picture/temp2.png'
default_t3 = '/home/iiot-b20/Documents/Robotvision/flet-camera-app/image_comppressor_picture/temp3.png'
template1_path = os.environ.get('TM_TEMPLATE1', os.environ.get('TM_TEMPLATE', default_t1))
template2_path = os.environ.get('TM_TEMPLATE2', default_t2)
template3_path = os.environ.get('TM_TEMPLATE3', default_t3)
source_path = os.environ.get('TM_SOURCE', '/home/iiot-b20/Documents/Robotvision/flet-camera-app/image_comppressor_picture/')

# Load template 1
template1 = cv2.imread(template1_path, cv2.IMREAD_GRAYSCALE)
if template1 is None:
    print(f"Read template1 failed: {template1_path}")
    sys.exit(111)

# Initialize matcher1 and set template1
matcher1 = matcher
matcher1.set_template(template1)

# Optionally load template 2 and create matcher2
template2 = cv2.imread(template2_path, cv2.IMREAD_GRAYSCALE)
matcher2 = None
if template2 is None:
    print(f"Note: template2 not found or unreadable: {template2_path}. Proceeding with a single template.")
else:
    try:
        matcher2 = Matcher(dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea)
        matcher2.set_template(template2)
        print(f"Loaded template2: {template2_path}")
    except Exception as e:
        print(f"Warning: failed to create second matcher for template2: {e}. Proceeding with a single template.")

# Optionally load template 3 and create matcher3
template3 = cv2.imread(template3_path, cv2.IMREAD_GRAYSCALE)
matcher3 = None
if template3 is None:
    print(f"Note: template3 not found or unreadable: {template3_path}. Proceeding without template3.")
else:
    try:
        matcher3 = Matcher(dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea)
        matcher3.set_template(template3)
        print(f"Loaded template3: {template3_path}")
    except Exception as e:
        print(f"Warning: failed to create third matcher for template3: {e}. Proceeding without template3.")

def _draw_and_annotate(matcher_obj, src_img, matches_count, color=(0, 255, 0), filename_hint=None):
    out_img = src_img.copy()
    for i in range(min(matches_count, matcher_obj.maxCount)):
        result = matcher_obj.results[i]
        if result.score > 0:
            pts = np.array(
                [
                    [result.leftTopX, result.leftTopY],
                    [result.leftBottomX, result.leftBottomY],
                    [result.rightBottomX, result.rightBottomY],
                    [result.rightTopX, result.rightTopY],
                ],
                np.int32,
            )
            cv2.polylines(out_img, [pts], True, color, 2)
            cv2.putText(
                out_img,
                f"{result.score:.3f}",
                (int(result.leftTopX), int(result.leftTopY) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
    if filename_hint is not None:
        cv2.putText(out_img, filename_hint, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    return out_img

def _make_matcher_for_template(template_img):
    m = Matcher(dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea)
    m.set_template(template_img)
    return m

def _list_images_in_dir(dir_path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    items = []
    try:
        for name in sorted(os.listdir(dir_path)):
            _, ext = os.path.splitext(name)
            if ext.lower() in exts:
                items.append(os.path.join(dir_path, name))
    except Exception as e:
        print(f"Failed to list directory {dir_path}: {e}")
        return []
    return items

show = os.environ.get('TM_SHOW', '1') == '1'

if os.path.isdir(source_path):
    # Folder browsing mode
    img_list = _list_images_in_dir(source_path)
    if not img_list:
        print(f"No images found in folder: {source_path}")
        sys.exit(111)
    print(f"Found {len(img_list)} images in {source_path}. Use Left/Right arrows to navigate, q to quit, s to save current.")
    idx = 0
    win = 'result'
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        img_path = img_list[idx]
        src = cv2.imread(img_path)
        if src is None:
            print(f"Failed to read image: {img_path}")
            # Skip to next
            idx = (idx + 1) % len(img_list)
            continue
        # Run matching for template1
        matches1 = matcher1.match(src)
        if matches1 < 0:
            print("Match (template1) failed!")
            idx = (idx + 1) % len(img_list)
            continue
        out = _draw_and_annotate(matcher1, src, matches1, (0, 255, 0), os.path.basename(img_path))
        # Overlay current parameters
        cv2.putText(out, f"SCORE>={scoreThreshold:.2f}  ANGLE±{angle:.1f}  IOU<={iouThreshold:.2f}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Run matching for template2 if available
        if matcher2 is not None:
            matches2 = matcher2.match(src)
            if matches2 >= 0:
                # Overlay on the same output with a different color
                out = _draw_and_annotate(matcher2, out, matches2, (255, 0, 0))
            else:
                print("Match (template2) failed!")

        # Run matching for template3 if available
        if matcher3 is not None:
            matches3 = matcher3.match(src)
            if matches3 >= 0:
                # Use a distinct color (orange)
                out = _draw_and_annotate(matcher3, out, matches3, (0, 165, 255))
            else:
                print("Match (template3) failed!")

        # Print summary
        summary = [f"T1:{matches1}"]
        if matcher2 is not None:
            summary.append(f"T2:{matches2}")
        if matcher3 is not None:
            # matches3 may be unbound if matcher3 is None, guarded above. If matcher3 exists but we didn't set it due to read fail, it's None.
            try:
                summary.append(f"T3:{matches3}")
            except NameError:
                pass
        print(f"[{idx+1}/{len(img_list)}] {os.path.basename(img_path)} -> {' '.join(summary)}")
        if show:
            cv2.imshow(win, out)
            k = cv2.waitKeyEx(0)
            # keys: left (81 or 2424832), right (83 or 2555904), q (113) or ESC(27)
            if k in (ord('q'), 27):
                break
            elif k in (81, 2424832):  # left
                idx = (idx - 1 + len(img_list)) % len(img_list)
            elif k in (83, 2555904):  # right
                idx = (idx + 1) % len(img_list)
            elif k in (ord('+'), ord('=')):
                # Increase score threshold
                scoreThreshold = min(scoreThreshold + 0.05, 0.99)
                print(f"New scoreThreshold: {scoreThreshold:.2f}")
                matcher1 = _make_matcher_for_template(template1)
                if matcher2 is not None and template2 is not None:
                    matcher2 = _make_matcher_for_template(template2)
                if matcher3 is not None and template3 is not None:
                    matcher3 = _make_matcher_for_template(template3)
                continue  # reprocess same image with new params
            elif k in (ord('-'), ord('_')):
                # Decrease score threshold
                scoreThreshold = max(scoreThreshold - 0.05, 0.0)
                print(f"New scoreThreshold: {scoreThreshold:.2f}")
                matcher1 = _make_matcher_for_template(template1)
                if matcher2 is not None and template2 is not None:
                    matcher2 = _make_matcher_for_template(template2)
                if matcher3 is not None and template3 is not None:
                    matcher3 = _make_matcher_for_template(template3)
                continue
            elif k == ord(']'):
                # Increase angle range
                angle = min(angle + 5.0, 180.0)
                print(f"New angle (±): {angle:.1f}")
                matcher1 = _make_matcher_for_template(template1)
                if matcher2 is not None and template2 is not None:
                    matcher2 = _make_matcher_for_template(template2)
                if matcher3 is not None and template3 is not None:
                    matcher3 = _make_matcher_for_template(template3)
                continue
            elif k == ord('['):
                # Decrease angle range
                angle = max(angle - 5.0, 0.0)
                print(f"New angle (±): {angle:.1f}")
                matcher1 = _make_matcher_for_template(template1)
                if matcher2 is not None and template2 is not None:
                    matcher2 = _make_matcher_for_template(template2)
                if matcher3 is not None and template3 is not None:
                    matcher3 = _make_matcher_for_template(template3)
                continue
            elif k in (ord('s'), ord('S')):
                # Save current result alongside image
                base = os.path.splitext(os.path.basename(img_path))[0]
                out_name = f"{base}_match.png"
                out_path = os.path.join(os.path.dirname(img_path), out_name)
                cv2.imwrite(out_path, out)
                print(f"Saved: {out_path}")
            else:
                # Any other key: move next
                idx = (idx + 1) % len(img_list)
        else:
            # Headless mode: just process all sequentially
            idx = (idx + 1) % len(img_list)
            if idx == 0:
                break
    if show:
        cv2.destroyAllWindows()
else:
    # Single-image mode with interactive tuning
    src = cv2.imread(source_path)
    if src is None:
        print(f"Read source image failed: {source_path}")
        sys.exit(111)

    win = 'result'
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        # Run with current params
        matches1 = matcher1.match(src)
        if matches1 < 0:
            print("Match (template1) failed!")
            break
        out = _draw_and_annotate(matcher1, src, matches1, (0, 255, 0))
        if matcher2 is not None:
            matches2 = matcher2.match(src)
            if matches2 >= 0:
                out = _draw_and_annotate(matcher2, out, matches2, (255, 0, 0))
        if matcher3 is not None:
            matches3 = matcher3.match(src)
            if matches3 >= 0:
                out = _draw_and_annotate(matcher3, out, matches3, (0, 165, 255))

        # Overlay current parameters
        cv2.putText(out, f"SCORE>={scoreThreshold:.2f}  ANGLE±{angle:.1f}  IOU<={iouThreshold:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save (always) latest result file
        result_file = os.environ.get('TM_RESULT', 'match_result.png')
        cv2.imwrite(result_file, out)

        if show:
            cv2.imshow(win, out)
            k = cv2.waitKeyEx(0)
            if k in (ord('q'), 27):
                break
            elif k in (ord('s'), ord('S')):
                cv2.imwrite(result_file, out)
                print(f"Saved: {result_file}")
            elif k in (ord('+'), ord('=')):
                scoreThreshold = min(scoreThreshold + 0.05, 0.99)
                print(f"New scoreThreshold: {scoreThreshold:.2f}")
                matcher1 = _make_matcher_for_template(template1)
                if matcher2 is not None and template2 is not None:
                    matcher2 = _make_matcher_for_template(template2)
                if matcher3 is not None and template3 is not None:
                    matcher3 = _make_matcher_for_template(template3)
                continue
            elif k in (ord('-'), ord('_')):
                scoreThreshold = max(scoreThreshold - 0.05, 0.0)
                print(f"New scoreThreshold: {scoreThreshold:.2f}")
                matcher1 = _make_matcher_for_template(template1)
                if matcher2 is not None and template2 is not None:
                    matcher2 = _make_matcher_for_template(template2)
                if matcher3 is not None and template3 is not None:
                    matcher3 = _make_matcher_for_template(template3)
                continue
            elif k == ord(']'):
                angle = min(angle + 5.0, 180.0)
                print(f"New angle (±): {angle:.1f}")
                matcher1 = _make_matcher_for_template(template1)
                if matcher2 is not None and template2 is not None:
                    matcher2 = _make_matcher_for_template(template2)
                if matcher3 is not None and template3 is not None:
                    matcher3 = _make_matcher_for_template(template3)
                continue
            elif k == ord('['):
                angle = max(angle - 5.0, 0.0)
                print(f"New angle (±): {angle:.1f}")
                matcher1 = _make_matcher_for_template(template1)
                if matcher2 is not None and template2 is not None:
                    matcher2 = _make_matcher_for_template(template2)
                if matcher3 is not None and template3 is not None:
                    matcher3 = _make_matcher_for_template(template3)
                continue
            else:
                # Any other key: just refresh
                continue
        else:
            # Headless mode, single pass
            print("Processed single image in headless mode.")
            break

    if show:
        cv2.destroyAllWindows()
