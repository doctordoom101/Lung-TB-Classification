import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# 1. Load model .keras menggunakan tf.keras, bukan Interpreter TFLite
MODEL_PATH = './model/mobilenetv2/best_mobilenetv2_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

IMG_HEIGHT = 224
IMG_WIDTH = 224

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Buat model yang memetakan gambar input ke aktivasi layer konvensional terakhir
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Hitung gradien untuk kelas prediksi teratas
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # Ambil probabilitas TB (indeks 0 atau sesuai output sigmoid Anda)
        class_channel = preds[:, 0]

    # Ambil gradien output terhadap feature map layer terakhir
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vektor rata-rata gradien (bobot intensitas)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Kalikan setiap channel feature map dengan "seberapa penting channel itu"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisasi heatmap antara 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def predict_image(image_input):
    # --- Preprocessing ---
    img = Image.fromarray(image_input).convert('RGB')
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_input_batch = np.expand_dims(img_array, axis=0)

    # --- Inference ---
    preds = model.predict(img_input_batch)
    tuberculosis_prob = float(preds[0][0])
    normal_prob = 1 - tuberculosis_prob

    # --- Grad-CAM Masking ---
    # Untuk MobileNetV2, layer konvensional terakhir biasanya bernama 'Conv_1' atau 'out_relu'
    # Jika error, cek dengan model.summary()
    try:
        last_conv_layer = "Conv_1" 
        heatmap = make_gradcam_heatmap(img_input_batch, model, last_conv_layer)
        
        # Rescale heatmap ke ukuran asli (0-255)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Gabungkan (Overlay) dengan gambar asli
        original_img = np.uint8(255 * img_array)
        heatmap = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
        
        # Berikan warna hanya jika probabilitas TB di atas threshold (misal 0.4)
        if tuberculosis_prob > 0.4:
            visualized_image = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        else:
            visualized_image = original_img # Tetap tampilkan ori jika normal
            
    except Exception as e:
        print(f"Grad-CAM Error: {e}. Menampilkan gambar original.")
        visualized_image = np.uint8(255 * img_array)

    result_labels = {
        "Normal": normal_prob,
        "Tuberculosis": tuberculosis_prob
    }

    return result_labels, visualized_image

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray Image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction Result"),
        gr.Image(label="Grad-CAM Diagnostic Visualization")
    ],
    title="TB Detection - Professional Diagnostic Tool",
    description="Sistem AI untuk mendeteksi Tuberculosis dengan visualisasi area patologi (Grad-CAM)."
)

iface.launch(debug=True)