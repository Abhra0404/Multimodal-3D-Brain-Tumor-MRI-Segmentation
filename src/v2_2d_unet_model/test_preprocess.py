import time

from v2_2d_unet_model.preprocess_v2 import load_patient


patient_id = "BraTS-GLI-01250-000"

selected_slices = [
    2, 11, 14, 22,
    36, 56, 59, 67,
    75, 79, 85, 103,
    105, 112, 122, 129
]

start = time.time()

images, masks = load_patient(
    patient_id,
    selected_slices
)

elapsed = time.time() - start

print("Images:", images.shape)
print("Masks:", masks.shape)
print("Images dtype:", images.dtype)
print("Masks dtype:", masks.dtype)
print("Time:", round(elapsed, 2), "seconds")