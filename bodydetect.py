from cv2 import circle
import mediapipe as mp
import cv2 as cv
mphands=mp.solutions.hands
hands=mphands.Hands()
mppose=mp.solutions.pose
pose=mppose.Pose()
mpface=mp.solutions.face_detection
face=mpface.FaceDetection()
facemesh=mp.solutions.face_mesh
mesh=facemesh.FaceMesh()
mpdraw=mp.solutions.drawing_utils
drawspecs=mpdraw.DrawingSpec(thickness=1,circle_radius=2)
cap=cv.VideoCapture(0)
while True:
    ret,Frame=cap.read()
    rgbframe=cv.cvtColor(Frame,cv.COLOR_BGR2RGB)
    handresults=hands.process(rgbframe)
    poseresults=pose.process(rgbframe)
    meshresults=mesh.process(rgbframe)
    if handresults.multi_hand_landmarks:
        for lns in handresults.multi_hand_landmarks:
            mpdraw.draw_landmarks(Frame,lns,mphands.HAND_CONNECTIONS)
    if poseresults.pose_landmarks:
        mpdraw.draw_landmarks(Frame,poseresults.pose_landmarks,mppose.POSE_CONNECTIONS)
    results=face.process(rgbframe)
    if results.detections:
        for id,ln in enumerate(results.detections):
            mpdraw.draw_detection(Frame,ln)
    if meshresults.multi_face_landmarks:
        for j,fln in enumerate(meshresults.multi_face_landmarks):
            mpdraw.draw_landmarks(Frame,fln,facemesh.FACEMESH_TESSELATION,drawspecs)
    cv.imshow('frame',Frame)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break