import matching as mt
import cv2
import os

# Example paths (update to your images)
template = '/home/iiot-b20/Documents/Robotvision/flet-camera-app/image_comppressor_picture/temp1.jpg'
source = '/home/iiot-b20/Documents/Robotvision/flet-camera-app/image_comppressor_picture/Image_10.png'

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
params = mt.MatchingParams(maxCount=1, scoreThreshold=0.6, iouThreshold=0.8, angle=10.0)
matcher = mt.create_matcher_for_template(template_img, dll_path, params)

# Prepare an image list to navigate. If `source` is a directory, use it; otherwise use
# the directory containing the provided source file and start at that file.
def _list_images(dir_path: str):
	exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
	items = []
	try:
		for name in sorted(os.listdir(dir_path)):
			_, ext = os.path.splitext(name)
			if ext.lower() in exts:
				items.append(os.path.join(dir_path, name))
	except Exception as e:
		print(f"Failed to list images in {dir_path}: {e}")
	return items


if os.path.isdir(source):
	img_list = _list_images(source)
	start_idx = 0
else:
	src_dir = os.path.dirname(source) or '.'
	img_list = _list_images(src_dir)
	try:
		start_idx = img_list.index(source)
	except ValueError:
		# If exact path not found, try basename match
		base = os.path.basename(source)
		start_idx = next((i for i, p in enumerate(img_list) if os.path.basename(p) == base), 0)

if not img_list:
	raise SystemExit(f"No images found to browse in: {source}")

win = 'match_view'
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
idx = start_idx
while True:
	img_path = img_list[idx]
	src_img = cv2.imread(img_path)
	if src_img is None:
		print(f"Failed to read image: {img_path}")
		idx = (idx + 1) % len(img_list)
		continue

	count, results = mt.run_match(matcher, src_img)
	print(f"[{idx+1}/{len(img_list)}] {os.path.basename(img_path)} -> matches: {count}")
	out = mt.draw_results(src_img, results)
	cv2.imshow(win, out)

	k = cv2.waitKeyEx(0)
	# q or ESC to quit
	if k in (ord('q'), 27):
		break
	# left arrow
	elif k in (81, 2424832):
		idx = (idx - 1) % len(img_list)
	# right arrow
	elif k in (83, 2555904):
		idx = (idx + 1) % len(img_list)
	# save current visualization
	elif k in (ord('s'), ord('S')):
		out_name = os.path.splitext(os.path.basename(img_path))[0] + '_match.png'
		out_path = os.path.join(os.path.dirname(img_path), out_name)
		cv2.imwrite(out_path, out)
		print(f"Saved: {out_path}")
	else:
		# any other key: advance
		idx = (idx + 1) % len(img_list)

cv2.destroyAllWindows()
mt.release_matcher(matcher)