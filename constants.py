# This file contains the constant paths used in the project's scripts.
# When using these scripts, please make sure to modify the paths according to your preferences.

# Root path for the Cityscapes dataset
ROOT_PATH = 'dataset/cityscapes'

# Path for the real-world videos in the Cityscapes dataset
REAL_PATH = 'dataset/cityscapes/images'

# Path for the depth maps videos in the Cityscapes dataset
DEPTH_PATH = 'dataset/cityscapes/depth'

# Path for the denoised depth maps videos in the Cityscapes dataset
DENOISED_DEPTH_PATH = 'dataset/cityscapes/denoised_depth'

# Output directory for real-world video models
REAL_OUTDIR = 'models/cityscapes/real'

# Output directory for depth map videomodels
DEPTH_OUTDIR = 'models/cityscapes/depth'

# Output directory for generated samples
SAMPLES_OUTDIR = 'samples/cityscapes'

# Path for saving the depth model
DEPTH_MODEL_PATH = 'models/cityscapes/depth_model.pt'

# Path for saving the GD-VDM model
GD_VDM_MODEL_PATH = 'models/cityscapes/gdvdm_model.pt'
