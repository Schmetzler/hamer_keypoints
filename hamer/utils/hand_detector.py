import mediapipe as mp
import cv2

class HandDetector():
    def __init__(self, taskfile, mode="IMAGE"):
        # Create a hand landmarker instance with the image mode:
        self.__mode = mode
        if self.__mode == "VIDEO":
            mode = mp.tasks.vision.RunningMode.VIDEO
        else:
            mode = mp.tasks.vision.RunningMode.IMAGE
        
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=taskfile),
            running_mode=mode,
            num_hands=2
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.result = None
        self.__frameNr = 1
        
    def detect_bboxes(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = cv_img.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_img)
        if self.__mode == "VIDEO":
            self.result = self.detector.detect_for_video(mp_image, self.__frameNr)
            self.__frameNr += 1
        else:
            self.result = self.detector.detect(mp_image)
        
        bboxes = {"Right": [], "Left": []}
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
        
        return bboxes
