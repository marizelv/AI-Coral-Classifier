# Run this in Google Colab.
# Upload your trained .h5 model, then execute this script.
#
# Example:
#   /content/optimized_sgd_best_model.h5
#
# TensorFlow.js creates model.json plus binary weight shard files.

!pip -q install tensorflowjs

import os
from google.colab import files

uploaded = files.upload()
h5_file = next(iter(uploaded.keys()))

output_dir = "/content/tfjs_model"
os.makedirs(output_dir, exist_ok=True)

!tensorflowjs_converter --input_format=keras "$h5_file" "$output_dir"

print("Converted files:")
for name in sorted(os.listdir(output_dir)):
    print(name)

# Optional: download the converted model as a ZIP.
!cd /content && zip -qr tfjs_model.zip tfjs_model

files.download("/content/tfjs_model.zip")
