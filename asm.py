import os
import numpy as np
import streamlit as st
from skimage import io, transform, feature, color
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# Step 1: Define Thresholds
hog_threshold = 0.95  # Threshold for HOG feature comparison
color_threshold = 0.9  # Threshold for color histogram comparison

# Step 2: Load Reference Images and Extract Features
@st.cache_data
def load_reference_features(reference_folder):
    reference_features = {}
    for filename in os.listdir(reference_folder):
        if filename.lower().endswith((".jpg", ".png")):
            filepath = os.path.join(reference_folder, filename)
            # Load reference image in color
            reference_image = io.imread(filepath)
            # Convert reference image to grayscale
            reference_image_gray = color.rgb2gray(reference_image)

            # Resize reference image
            resized_image = transform.resize(reference_image_gray, (128, 128))
            hog_features = feature.hog(resized_image, pixels_per_cell=(16, 16))
            color_hist = np.histogram(reference_image_gray, bins=8, range=(0, 1))[0] / 128**2
            reference_features[filename] = (reference_image, reference_image_gray, hog_features, color_hist)
    return reference_features

def compare_with_reference(user_image, user_image_gray, user_features, user_color_hist, reference_features):
    match_found = False
    for ref_filename, (ref_image, ref_image_gray, ref_hog_features, ref_color_hist) in reference_features.items():
        # Check grayscale match
        if np.array_equal(user_image_gray, ref_image_gray):
            match_found = True
            st.markdown(f"Grayscale match found with **{ref_filename}**.")
            # Compare HOG features
            hog_similarity = cosine_similarity([ref_hog_features], [user_features])[0][0]
            color_similarity = np.sum(np.minimum(ref_color_hist, user_color_hist))

            if hog_similarity >= hog_threshold and color_similarity >= color_threshold:
                st.success(f"Image matches with **{ref_filename}**. OK")
                match_found = True

                # Display images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(ref_image, caption=f"Reference Image: {ref_filename}")
                with col2:
                    st.image(user_image, caption='User Image')

            else:
                st.warning(f"Image is similar to, but does not exactly match **{ref_filename}**.")
                # Display user image and reference image side by side

                col1, col2 = st.columns(2)
                with col1:
                    st.image(ref_image, caption=f"Reference Image: {ref_filename}")
                with col2:
                    st.image(user_image, caption='User Image')

                # Consider using a more advanced similarity measure here,
                # e.g., deep learning features or structural similarity measures.

        else:
            # No grayscale match, compare with OK.jpg
            if not match_found and ref_filename.lower() == "ok.jpg":
                ok_image = ref_image
                ok_image_gray = ref_image_gray
                ok_hog_features = ref_hog_features
                ok_color_hist = ref_color_hist

                # If the image doesn't match "OK.jpg", proceed to highlighting differences
                if not (np.array_equal(user_image_gray, ok_image_gray) and
                        hog_similarity >= hog_threshold and
                        color_similarity >= color_threshold):
                    st.error(f"Image does not match any reference image.")
                    st.info("Check Each Higlighted  Region Either their is Color difference / Fuse is Missing or Fuse is placed or wrong location")
                    # Highlight the differences between user image and OK.jpg
                    diff_image_rgb = highlight_mismatch(user_image, ok_image, user_image_gray, ok_image_gray)
                    # Display OK.jpg and the highlighted difference image side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(ok_image, caption='OK.jpg')
                    with col2:
                        st.image(diff_image_rgb, caption='Highlighted Differences')
                    break

            else:
                # Skip non-OK reference images if no grayscale match
                continue

    if not match_found:
        st.error("Not-Okay Image ")


def highlight_mismatch(user_image, ok_image, user_image_gray, ok_image_gray):
    # Resize the user image to match the dimensions of the reference image
    user_image_resized = transform.resize(user_image, ok_image.shape)

    # Convert resized user image and reference image to grayscale for difference comparison
    user_gray_resized = color.rgb2gray(user_image_resized)
    ok_gray = color.rgb2gray(ok_image)

    # Identify the areas of difference between the user image and the reference image
    diff_image = np.abs(user_gray_resized - ok_gray)

    # Threshold for highlighting the differences
    threshold = 0.1
    diff_image_thresholded = diff_image > threshold

    # Convert the difference image to uint8 for OpenCV visualization
    diff_image_uint8 = (diff_image_thresholded * 255).astype(np.uint8)

    # Find contours of the differences
    contours, _ = cv2.findContours(diff_image_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the largest contour (region of difference)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Ensure that the box stays within the image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, user_image.shape[1] - x)
    h = min(h, user_image.shape[0] - y)

    # Draw different colored rectangles around the region of difference
    highlighted_image = user_image.copy()
    cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Draw horizontal box
    cv2.rectangle(highlighted_image, (x, y + h // 2 - 5), (x + w, y + h // 2 + 5), (0, 0, 255), 2)

    # Draw vertical box
    cv2.rectangle(highlighted_image, (x + w // 2 - 5, y), (x + w // 2 + 5, y + h), (255, 0, 0), 2)

    # Draw left box
    cv2.rectangle(highlighted_image, (x, y), (x + 10, y + h), (255, 255, 0), 2)

    # Draw right box
    cv2.rectangle(highlighted_image, (x + w - 10, y), (x + w, y + h), (255, 255, 0), 2)

    return highlighted_image

if __name__ == "__main__":
    st.title("Image Similarity Checker")
    st.markdown("---")

    # Dynamic reference folder path for Streamlit Sharing
    reference_folder = os.path.join(os.path.dirname(__file__), "ONS1", "ONS", "reference")
    reference_features = load_reference_features(reference_folder)

    user_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if user_image is not None:
        user_image_array = io.imread(user_image)  # Load user image
        user_image_gray = color.rgb2gray(user_image_array)
        resized_user_image = transform.resize(user_image_gray, (128, 128))
        user_features = feature.hog(resized_user_image, pixels_per_cell=(16, 16))
        user_color_hist = np.histogram(user_image_gray, bins=8, range=(0, 1))[0] / 128**2

        if st.button("Check Similarity"):
            compare_with_reference(user_image_array, user_image_gray, user_features, user_color_hist, reference_features)
