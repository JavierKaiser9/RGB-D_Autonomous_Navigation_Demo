import openvino as ov
import tensorflow as tf

# =============== 2 INPUTS 1 OUTPUT TENSORFLOW MODEL TO OPENVINO MODEL====================
loaded_model = tf.keras.models.load_model(r".\Z_DUAL_MODELS\M14_NEW_lr0.00001_ep64_NO_TEST")
loaded_model = ov.convert_model(loaded_model, input=[(1,224,224,3),(1,224,224,1)])
ov.save_model(loaded_model, "M14_NEW_lr0.00001_ep64_NO_TEST.xml")