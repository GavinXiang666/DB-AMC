from camera import *

camera_index_list = [0,1]
camera_capture_frame = {}
for cap_index in camera_index_list:
    camera = Camera(camera_index=cap_index)
    image = camera.Get_Frame()
    camera_capture_frame[cap_index] = image

for index, show_image in camera_capture_frame.items():
    cv2.imshow(f"Camera {index}",show_image)

cv2.waitKey(0)
cv2.destroyAllWindows()