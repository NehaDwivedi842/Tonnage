import os
import numpy as np
import streamlit as st
from skimage import io, transform, feature, color
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# Step 1: Define Thresholds
hog_threshold =  0.95
color_threshold =  0.9 

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
            resized_image = transform.resize(reference_image, (128, 128))  # Resize the colored image
            hog_features = feature.hog(color.rgb2gray(resized_image), pixels_per_cell=(16, 16))  # Compute HOG features from grayscale
            color_hist = np.histogram(resized_image, bins=8, range=(0, 1))[0] / 128**2  # Compute color histogram from colored image
            reference_features[filename] = (reference_image, reference_image_gray, hog_features, color_hist)
    return reference_features

def compare_with_reference(user_image, user_image_gray, user_features, user_color_hist, reference_features):
    match_found = False
    for ref_filename, (ref_image, ref_image_gray, ref_hog_features, ref_color_hist) in reference_features.items():
        # Check grayscale match
        if np.array_equal(user_image_gray, ref_image_gray):
            match_found = True
            st.write(f"Match found with {ref_filename}.")
            # Compare HOG features with Euclidean distance
            hog_distance = np.linalg.norm(ref_hog_features - user_features)
            color_distance = np.linalg.norm(ref_color_hist - user_color_hist)

            if hog_distance <= hog_threshold and color_distance <= color_threshold:
                st.success(f"Image matches with {ref_filename}. OK")
                match_found = True
                
                # Display images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(ref_image, caption=f"Reference Image: {ref_filename}")
                with col2:
                    st.image(user_image, caption='User Image')
                
            else:
                st.write(f"Image does not match with {ref_filename}. Not-OK")
                # Highlight the areas of mismatch
                highlight_mismatch(user_image, ref_image, user_image_gray, ref_image_gray)
            break

    if not match_found:
        st.error("Image does not match with any reference image. ")
        if rotate_and_compare_ok(user_image):
            match_found = True

            
def rotate_and_compare_ok(user_image):
    ok_image_path = os.path.join(reference_folder, "OK.jpg")
    ok_image = io.imread(ok_image_path)
    ok_image_resized = transform.resize(ok_image, (128, 128))
    ok_hog_features = feature.hog(color.rgb2gray(ok_image_resized), pixels_per_cell=(16, 16))
    ok_color_hist = np.histogram(ok_image_resized, bins=8, range=(0, 1))[0] / 128**2

    rotation_angles = [0, 90, 180, 270, 360]

    for angle in rotation_angles:
        rotated_user_image = transform.rotate(user_image, angle, resize=True)
        rotated_user_image_resized = transform.resize(rotated_user_image, (128, 128))
        hog_features = feature.hog(color.rgb2gray(rotated_user_image_resized), pixels_per_cell=(16, 16))
        color_hist = np.histogram(rotated_user_image_resized, bins=8, range=(0, 1))[0] / 128**2

        hog_distance = np.linalg.norm(ok_hog_features - hog_features)
        color_distance = np.linalg.norm(ok_color_hist - color_hist)

        st.write(f"Rotation angle: {angle} degrees")

        if hog_distance <= hog_threshold and color_distance <= color_threshold:
            st.write(f"Image matches with OK.jpg after rotation. OK")
            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(ok_image, caption="OK.jpg")
            with col2:
                st.image(rotated_user_image, caption=f'Rotated User Image ({angle} degrees)')
        else:
            st.write(f"Image does not match with OK.jpg after rotation. Not-OK")
            # Display the images 
            col1, col2 = st.columns(2)
            with col1:
                st.image(ok_image, caption="OK.jpg")
            with col2:
                st.image(rotated_user_image, caption=f'Rotated User Image ({angle} degrees)')

    return False



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
        resized_user_image = transform.resize(user_image_array, (128, 128))  # Resize the colored user image
        user_features = feature.hog(color.rgb2gray(resized_user_image), pixels_per_cell=(16, 16))  # Compute HOG features from grayscale
        user_color_hist = np.histogram(resized_user_image, bins=8, range=(0, 1))[0] / 128**2  # Compute color histogram from colored image

        if st.button("Check Similarity"):
            compare_with_reference(user_image_array, user_image_gray, user_features, user_color_hist, reference_features)
