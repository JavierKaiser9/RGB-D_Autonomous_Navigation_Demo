import time
from realsense_camera_setup import *
import openvino as ov


def final_class(model_class):
    if model_class == 0:
        final_output = "Forward_Path"
    elif model_class == 1:
        final_output = "Forward_Path_Entry"
    elif model_class == 2:
        final_output = "Forward_Open_Area"
    elif model_class == 3:
        final_output = "Left_Obstacle"
    elif model_class == 4:
        final_output = "Left_Street"
    elif model_class == 5:
        final_output = "Left_Path_Limit"
    elif model_class == 6:
        final_output = "Left_Find_Path"
    elif model_class == 7:
        final_output = "Left_No_Path"
    elif model_class == 8:
        final_output = "Right_Obstacle"
    elif model_class == 9:
        final_output = "Right_Street"
    elif model_class == 10:
        final_output = "Right_Path_Limit"
    elif model_class == 11:
        final_output = "Right_Find"
    elif model_class == 12:
        final_output = "Right_No_Path"
    elif model_class == 13:
        final_output = "STOP"
    else:
        final_output = "NO CLASS"

    return final_output


def cnn_ov_control(loaded_model, cam, main_classes, output_layer):
    start = time.time()

    depth_img, color_img = cam.get_frame()
    depth_img[depth_img > 15000] = 15000

    # Depth normalization
    depth_frame = cv2.resize(depth_img, (224, 224))
    depth_frame = np.expand_dims(depth_frame, axis=0)
    depth_frame = np.expand_dims(depth_frame, axis=-1)
    # RGB normalization
    rgb_frame = cv2.resize(color_img, (224, 224))
    rgb_frame = np.expand_dims(rgb_frame, axis=0)

    # Predictions

    predictions = loaded_model((rgb_frame, depth_frame))[output_layer]
    predicted_class = main_classes[np.argmax(predictions)]
    movement = final_class(np.argmax(predictions))

    end = time.time()
    fps = 1. / (end - start)

    # Add FPS and predicted class label to the frame
    text = f"FPS: {fps:.3f} - Class: {movement}"
    text_class = f"Class: {predicted_class}"
    cv2.putText(color_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(depth_img, text_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show Results
    cv2.imshow('depth', depth_img)
    cv2.imshow('color', color_img)


# ===================== MODEL SETUP ===================================

core = ov.Core()
model_name = "M14_NEW_lr0.00001_ep32_NO_TEST"
model_xml = f"{model_name}.xml"
model = core.read_model(model=model_xml)
compiled_model = core.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)

cam = Camera(640, 30)

main_classes = ["c1_forward_PATH",
                "c2_forward_ENTER_PATH",
                "c3_forward_OPEN_AREA",
                "c4_left_OBSTACLE",
                "c5_left_STREET",
                "c6_left_PATH_TURN",
                "c7_left_FIND_PATH",
                "c8_left_NO_PATH",
                "c9_right_OBSTACLE",
                "c10_right_STREET",
                "c11_right_PATH_TURN",
                "c12_right_FIND_PATH",
                "c13_right_NO_PATH",
                "c14_STOP"]

# ===================== MAIN CONTROL LOOP ============================
while True:
    start = time.time()
    cnn_ov_control(compiled_model, cam, main_classes, output_layer)
    end = time.time()
    fps = 1. / (end - start)
    print("=========")
    print(f"Loop FPS :{fps}")
    print("=========")

    key = cv2.waitKey(1)  # Check if a key was pressed
    if key == 27:  # Press 'Esc' key to exit
        cam.final_camera()
        break
