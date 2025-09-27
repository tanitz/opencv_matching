import matching as mt
import cv2
import os
import math
import numpy as np


dll_path = mt.find_library_path()

def _first_center(centers):
    # centers can be a tuple (x,y) or a list of tuples; return (int(x), int(y)) or None
    if not centers:
        return None
    if isinstance(centers, (tuple, list)) and len(centers) == 2 and all(isinstance(v, (int, float)) for v in centers):
        return (int(centers[0]), int(centers[1]))
    # assume list-like of points
    try:
        c = centers[0]
        return (int(c[0]), int(c[1]))
    except Exception:
        return None

def _angle_between(p1, p2, p3):
    # returns signed angle degrees from vector p1->p2 to p1->p3 (-180..180)
    v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    u = np.array([p3[0] - p1[0], p3[1] - p1[1]], dtype=float)
    nv = np.linalg.norm(v)
    nu = np.linalg.norm(u)
    if nv == 0 or nu == 0:
        return None
    dot = float(np.dot(v, u))
    det = float(v[0] * u[1] - v[1] * u[0])  # 2D cross (z)
    ang = math.degrees(math.atan2(det, dot))
    return ang

def match_and_annotate(template1_img,
                       template2_img,
                       source_img,
                       dll_path=None,
                       params1=None,
                       params2=None,
                       draw_angle=True):

    if dll_path is None:
        dll_path = mt.find_library_path()

    if params1 is None:
        params1 = mt.MatchingParams(maxCount=1, scoreThreshold=0.6, iouThreshold=0.8, angle=5.0)
    if params2 is None:
        params2 = mt.MatchingParams(maxCount=1, scoreThreshold=0.4, iouThreshold=0.6, angle=1.0)

    matcher1 = mt.create_matcher_for_template(template1_img, dll_path, params1)
    matcher2 = mt.create_matcher_for_template(template2_img, dll_path, params2)

    count1, results1, center1 = mt.run_match(matcher1, source_img)
    count2, results2, center2 = mt.run_match(matcher2, source_img)

    c1 = _first_center(center1)
    c2 = _first_center(center2)
    c3 = None
    if c1 and c2:
        c3 = (int(c1[0]), int(c2[1]))

    # draw combined results
    image = source_img.copy()
    image = mt.draw_results(image, list(results1) + list(results2))

    if c1 and c2:
        cv2.line(image, c1, c2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(image, c1, 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, c2, 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)

        if draw_angle and c3:
            ang_signed = _angle_between(c1, c2, c3)
            if ang_signed is None:
                ang_text = "angle: n/a"
            else:
                ang_abs = abs(ang_signed)
                ang_text = f"angle: {ang_abs:.1f} deg (signed {ang_signed:.1f})"
            txt_pos = (int(c1[0] + 8), int(c1[1] - 10))
            cv2.putText(image, ang_text, txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(image, c3, 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.line(image, c3, c1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2.line(image, c3, c2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    meta = {
        "count1": count1, "results1": results1, "center1": center1,
        "count2": count2, "results2": results2, "center2": center2,
        "c1": c1, "c2": c2, "c3": c3
    }
    return image, meta