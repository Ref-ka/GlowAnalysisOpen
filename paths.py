from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

IMAGES_DIR = str(ROOT_DIR / "images")
LABELS_DIR = str(ROOT_DIR / "labels")
VIDEOS_DIR = str(ROOT_DIR / "videos")
MODELS_DIR = str(ROOT_DIR / "trained_models")
TRAIN_DATA_DIR = str(ROOT_DIR / "training_data")
