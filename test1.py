import matching as mt
import cv2
import os

# Example paths (update to your images)
template = '/home/iiot-b20/Documents/Robotvision/flet-camera-app/image_comppressor_picture/temp1.jpg'
source = '/home/iiot-b20/Documents/Robotvision/flet-camera-app/image_comppressor_picture/Image_27.png'

def _resolve_path(p: str) -> str:
	import os
	if os.path.exists(p):
		return p
	# Fix double extensions like .png.jpg or .jpg.png
	if p.endswith('.png.jpg'):
		alt = p[:-4]
		if os.path.exists(alt):
			return alt
	if p.endswith('.jpg.png'):
		alt = p[:-4]
		if os.path.exists(alt):
			return alt
	# Try swapping common extensions
	root, ext = os.path.splitext(p)
	for new_ext in ('.jpg', '.jpeg', '.png'):
		if new_ext.lower() == ext.lower():
			continue
		alt = root + new_ext
		if os.path.exists(alt):
			return alt
	return p

template = _resolve_path(template)
source = _resolve_path(source)

template_img = cv2.imread(template)
source_img = cv2.imread(source)
if template_img is None:
	raise SystemExit(f"Failed to read template image. Please verify the path: {template}")
if source_img is None:
	raise SystemExit(f"Failed to read source image. Please verify the path: {source}")


dll_path = mt.find_library_path()
params = mt.MatchingParams(maxCount=1, scoreThreshold=0.6, iouThreshold=0.8, angle=5.0)
matcher = mt.create_matcher_for_template(template_img, dll_path, params)
count, results = mt.run_match(matcher, source_img)
vis = mt.draw_results(source_img, results)
cv2.imshow('Matches', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
