import streamlit as st
import numpy as np
import tempfile
import os
from src.image_processing import align_images, load_image, resize_image
from src.change_detection import compute_difference
from src.visualization import create_heatmap

st.title("DeltaVision: Change Detection Tool")

st.markdown("Upload two images to detect changes between them.")

uploaded1 = st.file_uploader("Upload first image", type=["png", "jpg", "tif", "tiff"])
uploaded2 = st.file_uploader("Upload second image", type=["png", "jpg", "tif", "tiff"])

threshold = st.slider("Change Detection Threshold", min_value=0, max_value=255, value=30)

if uploaded1 and uploaded2:
    # Save uploaded files to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded1.name)[1]) as f1:
        f1.write(uploaded1.read())
        path1 = f1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded2.name)[1]) as f2:
        f2.write(uploaded2.read())
        path2 = f2.name

    try:
        # Load images
        image1 = load_image(path1)
        image2 = load_image(path2)

        st.image([image1, image2], caption=["First Image", "Second Image"], width=300)

        # Resize if shapes differ
        if image1.shape != image2.shape:
            image2 = resize_image(image2, (image1.shape[1], image1.shape[0]))
            st.info("Images resized to match dimensions.")

        # Align images
        aligned1, aligned2 = align_images(image1, image2)

        # Compute difference
        diff = compute_difference(aligned1, aligned2)

        # Create heatmap
        heatmap = create_heatmap(diff)

        # Display results
        st.image(heatmap, caption="Change Heatmap", width=600)

        # Calculate change percentage
        change_percentage = np.count_nonzero(diff > threshold) / diff.size * 100
        st.metric("Change Percentage", f"{change_percentage:.2f}%")

        st.write(f"Diff stats: Min={np.min(diff)}, Max={np.max(diff)}, Mean={np.mean(diff):.2f}")

    except Exception as e:
        st.error(f"Error processing images: {str(e)}")
    finally:
        # Clean up temp files
        os.unlink(path1)
        os.unlink(path2)

st.markdown("---")
st.markdown("Built with Streamlit and OpenCV.")