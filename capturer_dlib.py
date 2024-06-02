import cv2
from time import sleep
import dlib
import uuid

def catch_pic_from_video(config: dict) -> None:
    cap = cv2.VideoCapture(config['camera_idx'])
    detector = dlib.get_frontal_face_detector()  

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = detector(grey, 1)  

        if len(face_rects) > 0:
            for face_rect in face_rects:
                x, y, w, h = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
                img_name = f"{config['path_name']}/{uuid.uuid1()}.jpg"
                image = frame[y-10:y+h+10, x-10:x+w+10]
                cv2.imwrite(img_name, image)
                config['num'] += 1
                if config['num'] > config['catch_pic_num']:
                    break
                cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"num:{config['num']}", (x+30, y+30), font, 1, (255, 0, 255), 4)

        if config['num'] > config['catch_pic_num']:
            break

        cv2.imshow(config['window_name'], frame)
        c = cv2.waitKey(10)

        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = {
        'num': 0,
        'camera_idx': 0,
        'catch_pic_num': 100,
        'path_name': './data/anchor',
        'window_name': "Say 'cheese!'",
        'scale_factor': 1.2,
        'minNeighbors': 3,
        'min_width': 112,
        'min_height': 112
    }
    catch_pic_from_video(config)
