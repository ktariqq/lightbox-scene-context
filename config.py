from pathlib import Path

# Project root and main directories.
ROOT = Path(__file__).resolve().parent
DATA, SUBSET, FEAT, RES, UAE, UAE_SUBSET = [
    ROOT / p for p in ("data", "subset", "features", "results", "uae", "uae_subset")
]
MEET10 = ROOT / "meet10"  # MEET test views degraded to 10 m (resolution control)
MAN, UMAN = ROOT / "subset_manifest.csv", ROOT / "uae_manifest.csv"

# Base reproducibility seed.
SEED = 42

# Per-class sample counts for the MEET subset.
N_TRAIN, N_VAL, N_TEST = 200, 50, 100  # per-class sample counts
SPLITS = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}

# 24 MEET classes; sorted order is the label index.
CLASSES = ["Airport", "Beach", "Brownfield", "Chemical plant", "Construction", "Dry field", "Highway", "Hills", "Landfill", "Military training ground", "Mosque", "Mountain", "Petroleum well", "Port", "Quarry", "Resort", "Rock", "Roundabout", "Sand", "Scrub", "Solar power plant", "Substation", "Swimming pool", "Thermal power plant"]
assert CLASSES == sorted(CLASSES)

# Three nested views and their crop boxes in the 1280x1280 image.
VIEWS = ["center", "surround", "global"]
BOXES = [(512, 512, 768, 768), (256, 256, 1024, 1024), (0, 0, 1280, 1280)]

# Plot colors.
PURPLE, NAVY = "#663399", "#0b1026"

# Size estimate for one original MEET JPEG.
EST_MB = 0.44  # avg MB per original jpg (Sand: 0.64 GB / 1460 images)

# UAE filtering and STAC settings.
MIN_UAE = 5  # min accepted UAE samples per class
RES_MAX = 1.5  # flag UAE source resolution above this (m); all Sentinel-2 samples are flagged by design
CLOUD_MAX = 5  # max scene cloud cover (%) in the STAC query
CLOUD_FLAG = 1.0  # scene cloud cover (%) at or above this sets blur_flag
S2_URL = "https://earth-search.aws.element84.com/v1"  # STAC API base
S2_COLL = "sentinel-2-c1-l2a"  # Sentinel-2 collection
S2_ASSET = "visual"  # fallback collection: sentinel-2-l2a
S2_DATES = "2025-11-01T00:00:00Z/2026-03-31T23:59:59Z"  # winter window, least haze
S2_GSD = 10  # Sentinel-2 pixel size in meters
NPX = 128  # 128 px x 10 m = 1280 m, same footprint as MEET

# UAE classes resolvable at 10 m.
UAE_CLASSES = ["Airport", "Beach", "Brownfield", "Chemical plant", "Construction", "Highway", "Landfill", "Mountain", "Port", "Quarry", "Resort", "Rock", "Sand", "Scrub", "Solar power plant", "Thermal power plant"]
N_UAE = 15  # max candidates per class (fewer to review)
UA = "aire410-course-project (contact: kommaltariq@gmail.com)"  # contact for the user agent
DELAY = 0.5  # seconds between Overpass/STAC requests

# Rough zip sizes by class, in GB.
ZIP_GB = {"Sand": .64, "Rock": .63, "Scrub": 8.97, "Dry field": 55.3, "Brownfield": .71, "Construction": 1.01, "Quarry": 3.28, "Landfill": .66, "Mountain": 1.33, "Hills": 1.12, "Petroleum well": .64, "Solar power plant": 21.9, "Thermal power plant": .70, "Chemical plant": .71, "Substation": .71, "Mosque": 1.02, "Resort": .67, "Swimming pool": 1.29, "Roundabout": .79, "Port": .75, "Airport": 1.61, "Military training ground": 1.21, "Highway": 2.77, "Beach": .85}

# Output model directory and statistical settings.
MODELS = ROOT / "models"
PRACT = 0.01  # practical-significance threshold on accuracy difference
NBOOT = 1000  # bootstrap resamples