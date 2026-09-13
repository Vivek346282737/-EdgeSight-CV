import os
import cv2
import json
import numpy as np

class DatasetCurationPipeline:
    """Automated dataset quality checks, synthetic data generation, and active learning loop."""
    def __init__(self, harvest_dir="data/hard_samples", export_dir="data/curated_dataset"):
        self.harvest_dir = harvest_dir
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def generate_synthetic_augmentations(self, image):
        """Simulates diverse industrial conditions: low-light, blur, and sensor noise."""
        augmented = []
        # 1. Low light condition (synthetic underexposure)
        low_light = cv2.convertScaleAbs(image, alpha=0.6, beta=-20)
        augmented.append(("low_light", low_light))

        # 2. Motion blur simulation
        kernel_size = 5
        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        kernel_v /= kernel_size
        blurred = cv2.filter2D(image, -1, kernel_v)
        augmented.append(("motion_blur", blurred))

        return augmented

    def curate_harvested_samples(self):
        if not os.path.exists(self.harvest_dir):
            print(f"[INFO] No harvest folder found at {self.harvest_dir}")
            return

        samples = [f for f in os.listdir(self.harvest_dir) if f.endswith(('.jpg', '.png'))]
        print(f"[MLOps] Processing {len(samples)} active-learning edge cases...")

        for sample_name in samples:
            img_path = os.path.join(self.harvest_dir, sample_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Synthetic variations export
            variations = self.generate_synthetic_augmentations(img)
            base_id = os.path.splitext(sample_name)[0]
            for tag, aug_img in variations:
                out_path = os.path.join(self.export_dir, f"{base_id}_{tag}.jpg")
                cv2.imwrite(out_path, aug_img)

        print(f"[SUCCESS] Synthetic dataset curated under: {self.export_dir}")

if __name__ == "__main__":
    curator = DatasetCurationPipeline()
    curator.curate_harvested_samples()
