import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from PIL import Image
from helper_functions import plot_loss_curves

# ===================== GPU MEMORY LIMIT CONFIGURATION =======================

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
    except RuntimeError as e:
        print(e)


# ============================ FUNCTIONS ======================================

def create_data(dir, main_categories, labels, mode):
    """
        Loads images from a specified directory, processes them, and assigns labels
        for training a TensorFlow model. Supports both grayscale and color image processing.
        This function is necessary to reduce the stress on the GPU RAM memery and fit the data to the model.

        Parameters:
        - dir (str): Path to the directory containing sub-folders with images.
        - main_categories (list): List of subdirectory names representing image categories.
        - labels (list): Corresponding labels for each category in `main_categories`.
        - mode (str): Image processing mode:
            - "gray": Loads images in grayscale, normalizes them, and rescales intensity.
            - Any other value: Loads images in color using OpenCV.

        Returns:
        - data (list): List of TensorFlow tensors representing images.
        - label (list): List of corresponding category labels.
    """

    label = []
    data = []
    IMG_SIZE = (224, 224)

    for category in main_categories:
        path = os.path.join(dir, category)
        for img in os.listdir(path):
            if mode == "gray":
                img_array = np.array(Image.open(os.path.join(path, img)), dtype=np.int32)
                norm_img = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
                res_norm_img = Image.fromarray((norm_img * 15000).astype(np.uint16)).resize(IMG_SIZE)
                new_img_array = np.array(res_norm_img, dtype=np.int32)
                new_img_array = np.expand_dims(new_img_array, axis=-1)
                img_array = tf.convert_to_tensor(new_img_array)

            else:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, IMG_SIZE)
                img_array = np.array(img_array)
                img_array = tf.convert_to_tensor(img_array)
            data.append(img_array)
            label.append(labels[main_categories.index(category)])

    return data, label


def normalize_rgb_depth_image(image_rgb, image_d, label):
    """Creates tuples of the 3 elements for training the 2 inputs 1 output model."""

    return ((image_rgb, image_d), label)


# ====================== MAIN VARIABLES ========================

IMG_SIZE = (224, 224)
IMG_DIM_RGB = (224, 224, 3)
IMG_DIM_DEPTH = (224, 224, 1)
BATCH_SIZE_TRIPLE = 12
main_categories = ["c1_forward_PATH",
                   "c2_forward_PATH_ENTRY",
                   "c3_forward_OPEN_AREA",
                   "c4_left_OBSTACLE",
                   "c5_left_STREET",
                   "c6_left_PATH_LIMIT",
                   "c7_left_FIND_PATH",
                   "c8_left_NO_PATH",
                   "c10_right_STREET",
                   "c11_right_PATH_LIMIT",
                   "c12_right_FIND_PATH",
                   "c13_right_NO_PATH",
                   "c14_STOP"]
NUM_OF_CATEGORIES = len(main_categories)

# Encode the previous label into one-hot type
encoded_labels = tf.one_hot(np.arange(NUM_OF_CATEGORIES), depth=len(main_categories))

# ======================== DATA LOADING ==========================

# Main folders of the data (NEED TO BE SET ACCORDING TO THE USER)
rgb_train_dir = r".\X_NEW_RGB\train"
depth_train_dir = r".\X_NEW_DEPTH\train"
rgb_test_dir = r".\X_NEW_RGB\test"
depth_test_dir = r".\X_NEW_DEPTH\test"
# Save the data into list as Tensors for using in the GPU
rgb_train_img, rgb_train_labels = create_data(rgb_train_dir, main_categories, encoded_labels, "rgb")
depth_train_img, depth_train_labels = create_data(depth_train_dir, main_categories, encoded_labels, "gray")
rgb_test_img, rgb_test_labels = create_data(rgb_test_dir, main_categories, encoded_labels, "rgb")
depth_test_img, depth_test_labels = create_data(depth_test_dir, main_categories, encoded_labels, "gray")

# ============= MAKING TRIPLE DATA-SETS FOR TRAINING ============
train_dataset = tf.data.Dataset.from_tensor_slices((rgb_train_img, depth_train_img, rgb_train_labels))

# Mapping for optimization
train_dataset = train_dataset.map(normalize_rgb_depth_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache()

# Setting changes
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.batch(BATCH_SIZE_TRIPLE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
#
# ============= MAKING TRIPLE DATA-SETS FOR TESTING ============
test_dataset = tf.data.Dataset.from_tensor_slices((rgb_test_img, depth_test_img, rgb_test_labels))

# Mapping for optimization
test_dataset = test_dataset.map(normalize_rgb_depth_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE_TRIPLE)
test_dataset = test_dataset.cache()

# Setting changes
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# =========== EFFICIENTNET MODELS FOR RGB and DEPTH =============

# RGB MODEL
rgb_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    weights=None,
    input_shape=IMG_DIM_RGB,
    include_preprocessing=True,
)
rgb_model.trainable = True

# Depth MODEL
depth_model = tf.keras.applications.EfficientNetV2B1(
    include_top=False,
    weights=None,
    input_shape=IMG_DIM_DEPTH,
    include_preprocessing=False,
)
depth_model.trainable = True

# ==================== FULL MODEL DESIGN ======================

input_rgb = tf.keras.layers.Input(shape=IMG_DIM_RGB, name='input_rgb')
input_depth = tf.keras.layers.Input(shape=IMG_DIM_DEPTH, name='input_depth')

# Special pre-processing for the Depth Images
input_depth = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 7500, offset=-1)(input_depth)
x_rgb = rgb_model(input_rgb)
x_depth = depth_model(input_depth)

merged = tf.keras.layers.Concatenate()([x_rgb, x_depth])
merged = tf.keras.layers.GlobalAvgPool2D()(merged)
merged = tf.keras.layers.Flatten()(merged)
output = tf.keras.layers.Dense(NUM_OF_CATEGORIES, activation='softmax', name='output')(merged)

full_model = tf.keras.models.Model(inputs=[input_rgb, input_depth], outputs=output)

# =================== MODEL COMPILE ========================
full_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                   metrics=["accuracy"])

print(full_model.summary())  # If needed to see the whole structure of the model

# =================== MODEL FIT ========================
model_history = full_model.fit(train_dataset,
                               epochs=4,
                               batch_size=BATCH_SIZE_TRIPLE,
                               steps_per_epoch=len(train_dataset),
                               validation_data=test_dataset,
                               validation_steps=len(test_dataset),
                               )

# =================== MODEL EVALUATION ========================
plot_loss_curves(model_history)
model_results_1 = full_model.evaluate(test_dataset)

# =================== SAVING THE MODEL ========================
model_path = r".\Z_DUAL_MODELS\NAME_OF_THE_MODEL"  # User establish the path and the name of the Model

# Check if the model file exists
if os.path.exists(model_path):
    # Delete the existing model file
    os.remove(model_path)
else:
    print("No model before")

# Save the new model
full_model.save(model_path)

# =================== MODEL PERFORMANCE (CONSOLE DISPLAY) ========================

print("CREATED MODEL:")
print(f"Accuracy of the Dual model is : {model_results_1[1] * 100} %")
plt.show()
