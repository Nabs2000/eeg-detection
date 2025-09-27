import kagglehub

# Download latest version
path = kagglehub.dataset_download("harunshimanto/epileptic-seizure-recognition")

print("Path to dataset files:", path)