import streamlit as st
from detection_page import DetectionPage
import os

# Buat Landing Page
def LandingPage ():
    st.subheader("Apa itu YOLOv8 Person Counter?")

    # Paragraf penjelasan sistem
    st.write(
        f'<div style="width: 710px; text-align: justify;">'
        f'YOLOv8 Person Counter menggunakan model deteksi object YOLOv8 untuk mendeteksi keberadaan manusia dalam sebuah gambar atau video dan menghitung jumlah manusia yang terdeteksi'
        f'</div>',
        unsafe_allow_html=True
    )

    st.subheader("Tutorial Penggunaan")

    # Paragraf penjelasan cara pemakaian YOLOv8 Person Counter
    st.write(
        f'<div style="width: 710px; text-align: justify;">'
        f'<p>1. Upload gambar atau video yang ingin dilakukan proses deteksi manusia</p>'
        f'<p>2. Setelah file diunggah, tekan tombol proses untuk memulai proses deteksi</p>'
        f'<p>3. Setelah proses deteksi selesai, hasil akan ditampilkan dan dapat diunduh</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.subheader("Deteksi Gambar")

    # Membuat path ke file gambar
    image_path1 = os.path.join(os.path.dirname(__file__), "ruang.jpg")
    image_path2 = os.path.join(os.path.dirname(__file__), "ruang_detected.jpg")

    # Membuat 2 kolom untuk me-render elemen
    col1, col2 = st.columns(2)

    # Membuat tampilan contoh deteksi gambar
    with col1:
        st.markdown("##### Gambar Asli")
        st.image(image_path1, use_column_width=True)
    with col2:
        st.markdown("##### Gambar Hasil Deteksi")
        st.image(image_path2, use_column_width=True)

    # Membuat tampilan contoh deteksi video
    st.subheader("Deteksi Video")

    # Membuat 2 kolom untuk me-render elemen
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Video Asli")
    with col2:
        st.markdown("##### Video Hasil Deteksi")

    video_path = os.path.join(os.path.dirname(__file__), "contoh_deteksi_video.mp4")

    st.video(video_path)

def main():
    # Judul Aplikasi
    st.title("Welcome to YOLOv8 Person Counter")

    # Membuat sidebar untuk navigasi
    with st.sidebar:
        st.title("Menu")
        if st.button("Beranda"):
            st.session_state.page = "Home"
        if st.button("Deteksi"):
            st.session_state.page = "Object Detection"

    # Display halaman sesuai tombol yang ditekan
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    if st.session_state.page == "Home":
        LandingPage()
    elif st.session_state.page == "Object Detection":
        DetectionPage()

if __name__ == "__main__":
    main()