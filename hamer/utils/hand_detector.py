import mediapipe as mp
import cv2

def expand_pose_bbox(x,y, w, h, factor=2.5):
    x0 = x[0]
    y0 = y[0]

    for i in range(1,len(x)):
        x[i] = x0 + (x[i] - x0) * factor
        y[i] = y0 + (y[i] - y0) * factor

    return [max(min(x),0)*w, max(min(y),0)*h, min(max(x),1)*w, min(max(y),1)*h]

def check_box(x,y):
    return max(x) <= 1 and max(y) <= 1 and min(x) >= 0 and max(x) >= 0

class HandDetector():
    def __init__(self, taskfiles, mode="IMAGE"):
        # Create a hand landmarker instance with the image mode:
        self.__mode = mode
        if self.__mode == "VIDEO":
            mode = mp.tasks.vision.RunningMode.VIDEO
        else:
            mode = mp.tasks.vision.RunningMode.IMAGE

        self.detectors = {}
        if "hand" in taskfiles:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=taskfiles["hand"]),
                running_mode=mode,
                num_hands=2
            )
            self.detectors["hand"] = mp.tasks.vision.HandLandmarker.create_from_options(options)
        if "pose" in taskfiles:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=taskfiles["pose"]),
                running_mode=mode
            )
            self.detectors["pose"] = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.result = None
        self.__frameNr = 1

    def detect_bboxes(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = cv_img.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_img)

        bboxes = {}
        self.__frameNr = (self.__frameNr % 65535) + 1

        if "hand" in self.detectors:
            detector = self.detectors["hand"]
            if self.__mode == "VIDEO":
                self.result = detector.detect_for_video(mp_image, self.__frameNr)
            else:
                self.result = detector.detect(mp_image)

            for i, elem in enumerate(self.result.handedness):
                name = elem[0].category_name
                score = elem[0].score

                x, y, z = [],[],[]
                for landmark in self.result.hand_landmarks[i]:
                    x.append(landmark.x)
                    y.append(landmark.y)
                    z.append(landmark.z)

                bbox = [min(x)*w, min(y)*h, max(x)*w, max(y)*h]
                bboxes[name] = bbox

        if not "Right" in bboxes or not "Left" in bboxes:
            if "pose" in self.detectors:
                detector = self.detectors["pose"]
                if self.__mode == "VIDEO":
                    self.result = detector.detect_for_video(mp_image, self.__frameNr)
                else:
                    self.result = detector.detect(mp_image)

                x_r, x_l, y_r, y_l = [], [], [], []
                for i, landmark in enumerate(self.result.pose_landmarks[0]):
                    if i in [16,18,20,22]:
                        x_r.append(landmark.x)
                        y_r.append(landmark.y)
                    if i in [15,17,19,21]:
                        x_l.append(landmark.x)
                        y_l.append(landmark.y)

                if not "Right" in bboxes and check_box(x_r,y_r):
                    bboxes["Right"] = expand_pose_bbox(x_r, y_r, w, h)
                if not "Left" in bboxes and check_box(x_l, y_l):
                    bboxes["Left"] = expand_pose_bbox(x_l, y_l, w, h)

        return bboxes
