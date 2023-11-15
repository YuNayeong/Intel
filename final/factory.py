"""Smart Factory"""

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
import openvino as ov

from iotdemo import FactoryController
from iotdemo import MotionDetector
from iotdemo import ColorDetector

FORCE_STOP = False

def thread_cam1(q):
    # MotionDetector
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')

    # Load and initialize OpenVINO
    core = ov.Core()
    model_path = 'resources/openvino.xml'
    model = core.read_model(model_path)
    ppp = ov.preprocess.PrePostProcessor(model)
    ppp.input().tensor() \
        .set_shape((1, 224, 224, 3)) \
        .set_element_type(ov.Type.u8) \
        .set_layout(ov.Layout('NHWC'))
    ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
    ppp.input().model().set_layout(ov.Layout('NCHW'))
    ppp.output().tensor().set_element_type(ov.Type.f32)
    model = ppp.build()
    compiled_model = core.compile_model(model, 'CPU')

    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    # pylint: disable=E1101
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))

        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue
        input_tensor = np.expand_dims(detected, 0)

        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detected))

        # Inference OpenVINO
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        print(f"{probs}")

        # Calculate ratios
        x_ratio = probs[0]*100
        circle_ratio = probs[1]*100
        print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # in queue for moving the actuator 1
        if x_ratio > 80:
            print('Not Good Item')
            q.put(('PUSH', 1))
        else:
            print('Good Item')

    cap.release()
    q.put(('DONE', None))

def thread_cam2(q):
    # MotionDetector
    det = MotionDetector()
    det.load_preset('resources/motion.cfg', 'default')

    # ColorDetector
    color = ColorDetector()
    color.load_preset('resources/color.cfg', 'default')

    # HW2 Open "resources/conveyor.mp4" video clip
    # pylint: disable=E1101
    cap = cv2.VideoCapture('resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame))

        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue

        # Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 detected', detected))

        # Detect color
        predict = color.detect(detected)

        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio*100
        print(f"{name}: {ratio:.2f}%")

        # Enqueue to handle actuator 2
        if name == 'blue':

            q.put(('PUSH', 2))

    cap.release()
    q.put(('DONE', None))

def imshow(title, frame, pos=None):
    # pylint: disable=E1101
    cv2.namedWindow(title)
    if pos:
        # pylint: disable=E1101
        cv2.moveWindow(title, pos[0], pos[1])
    # pylint: disable=E1101
    cv2.imshow(title, frame)

def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # HW2 Create a Queue
    q = Queue()

    # HW2 Create thread_cam1 and thread_cam2 threads and start them.
    thread1 = threading.Thread(target=thread_cam1, args=(q,))
    thread2 = threading.Thread(target=thread_cam2, args=(q,))

    thread1.start()
    thread2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            # pylint: disable=E1101
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
            # HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = q.get(timeout=1)
            except Empty:
                continue

            name, frame = event
            # HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if name.startswith("VIDEO:"):
                imshow(name[6:], frame)

            # Control actuator, name == 'PUSH'
            elif name == 'PUSH':
                ctrl.push_actuator(frame)
            elif name == 'DONE':
                FORCE_STOP = True

            q.task_done()

    # thread finish
    thread1.join()
    thread2.join()

    # pylint: disable=E1101
    cv2.destroyAllWindows()
    ctrl.system_stop()
    ctrl.close()

    if __name__ == '__main__':
        try:
            main()
        except Exception:
            os._exit()
