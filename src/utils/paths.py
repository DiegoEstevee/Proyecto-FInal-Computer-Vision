from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
TEMPLATES_DIR = DATA_DIR / "templates"
CALIB_DIR = DATA_DIR / "calibration"

RESULTS_DIR = REPO_ROOT / "results"
REPORT_DIR = REPO_ROOT / "report"


CALIB_PARAMS = REPO_ROOT / "src" / "calibration" / "camera_params.npz"
