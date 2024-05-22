import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import cv2
import io
    
# Fungsi untuk menangani file yang diupload
def file_upload():
    # Tampilkan tombol unggah file
    uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "mp4"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]  # Dapatkan tipe file
        st.write(f"File berhasil diupload")

        # Tampilkan gambar atau video yang diunggah
        if file_type == "jpg" or file_type == "jpeg":
            st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True) # Tampilkan gambar yang diupload
            # Tombol untuk proses gambar jika jenis file adalah gambar
            if st.button("Proses Gambar"):
                process_image(uploaded_file)
        elif file_type == "mp4":
            st.video(uploaded_file) # Tampilkan video yang diupload
            # Tombol untuk proses video jika jenis file adalah video
            if st.button("Proses Video"):
                # Placeholder untuk elemen pemrosesan video
                show_video_results_button = st.empty()
                result_placeholder = st.empty()  # Placeholder frame hasil proses
                show_video_results_button.text("Video sedang diproses... (mungkin memerlukan waktu beberapa menit)")
                process_video(uploaded_file, show_video_results_button, result_placeholder)

# Fungsi untuk memproses gambar yang diunggah
def process_image(uploaded_file):
    # Muat model YOLOv8
    model = YOLO("yolov8n.pt")

    # Baca gambar sebagai array numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Lakukan deteksi pada gambar
    results = model(img, classes=0)  # Kelas 0 untuk objek orang

    # Nama kelas objek dalam model YOLOv8
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # Filter hasil hanya untuk kelas "person" (orang)
    person_count = 0
    for result in results:
        boxes = result.boxes  # Ambil koordinat bounding box yang terdeteksi
        cls = boxes.cls.tolist()  # Ubah tensor menjadi list
        for class_index in cls:
            class_name = class_names[int(class_index)]
            if class_name == 'person':
                person_count += 1  # Hitung jumlah orang

    # Gambar bounding box pada objek orang yang terdeteksi
    for result in results:
        if result.names[0] == 'person':
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Tambahkan jumlah total orang yang terdeteksi pada frame
    cv2.putText(img, f"Jumlah orang terdeteksi: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Convert gambar ke bytes untuk ditampilkan
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()

    # Tampilkan gambar yang telah dideteksi
    st.image(img_bytes, channels="BGR")

    # Tombol untuk mengunduh gambar yang telah dideteksi
    st.download_button(label="Download Hasil Deteksi Gambar", data=img_bytes, file_name="detected_image.jpg", mime="image/jpeg")

# Fungsi untuk memproses video
def process_video(uploaded_file, show_video_results_button, result_placeholder):

    # Simpan file yang diunggah ke lokasi sementara
    temp_location = tempfile.NamedTemporaryFile(delete=False)
    temp_location.write(uploaded_file.read())
    temp_location.close()  # Tutup file

    # Muat model YOLOv8
    model = YOLO("yolov8n.pt")

    # Buka file video
    cap = cv2.VideoCapture(temp_location.name)

    # Dapatkan properti video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Buat objek VideoWriter untuk menulis video output
    output_video_path = "detected_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_video_path, fourcc, frame_fps, (frame_width, frame_height))

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # inisiasi variabel untuk menghitung jumah orang yang terdeteksi
    total_people = 0

    # Looping untuk setiap frame dalam video
    while cap.isOpened():
        # Baca satu frame dari video
        ret, frame = cap.read()

        if ret:
            # Jalankan proses deteksi objek pada frame
            results = model.track(frame, persist=True, classes=0)

            # Visualisasikan hasil pada frame
            annotated_frame = results[0].plot() # Method bawaan dari YOLOv8

            # Tambahkan teks jumlah orang yang terdeteksi pada bagian atas kiri frame
            text = f"Counter: {results[0].verbose().replace(',', '')}"
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Tampilkan frame yang telah diberi anotasi
            result_placeholder.image(annotated_frame, channels="BGR")

            # Tulis frame yang telah diberi anotasi ke video output
            output_video.write(annotated_frame)

        else:
            break

    # Release objek penangkapan video dan video output
    cap.release()
    output_video.release()

    # Sembunyikan pesan "Proses Video..."
    show_video_results_button.empty()

    # Tampilkan pesan bahwa proses video selesai
    st.write(f"Video selesai dideteksi!")

    # Hapus file sementara
    os.unlink(temp_location.name)

    # Tambahkan tombol unduh untuk video yang telah diproses
    with open(output_video_path, "rb") as file:
        video_bytes = file.read()
    st.download_button(label="Download Hasil Deteksi Video", data=video_bytes, file_name="detected_video.mp4",
                       mime="video/mp4")

# Halaman deteksi
def DetectionPage():
    st.subheader("Silahkan Upload Gambar atau Video")
    file_upload()
