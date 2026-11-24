from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA,SUBSET,FEAT,RES,UAE,UAE_SUBSET=[ROOT/p for p in ("data","subset","features","results","uae","uae_subset")]
MEET10=ROOT/"meet10"  # MEET test views degraded to 10 m (resolution control)
MAN,UMAN=ROOT/"subset_manifest.csv",ROOT/"uae_manifest.csv"
SEED=42
N_TRAIN,N_VAL,N_TEST=200,50,100  # EDIT: per-class sample counts
SPLITS={"train":N_TRAIN,"val":N_VAL,"test":N_TEST}
CLASSES=["Airport","Beach","Brownfield","Chemical plant","Construction","Dry field","Highway","Hills","Landfill","Military training ground","Mosque","Mountain","Petroleum well","Port","Quarry","Resort","Rock","Roundabout","Sand","Scrub","Solar power plant","Substation","Swimming pool","Thermal power plant"]
assert CLASSES==sorted(CLASSES)
VIEWS=["center","surround","global"]
BOXES=[(512,512,768,768),(256,256,1024,1024),(0,0,1280,1280)]
PURPLE,NAVY="#663399","#0b1026"
EST_MB=0.44  # EDIT: avg MB per original jpg (Sand: 0.64 GB / 1460 images)
MIN_UAE=5  # EDIT: min accepted UAE samples per class
RES_MAX=1.5  # EDIT: flag UAE source resolution above this (m); all Sentinel-2 samples are flagged by design
CLOUD_MAX=5  # EDIT: max scene cloud cover (%) in the STAC query
CLOUD_FLAG=1.0  # EDIT: scene cloud cover (%) at or above this sets blur_flag (kept name for schema compatibility; here it is a QC flag)
S2_URL="https://earth-search.aws.element84.com/v1"; S2_COLL="sentinel-2-c1-l2a"; S2_ASSET="visual"  # fallback collection: sentinel-2-l2a
S2_DATES="2025-11-01T00:00:00Z/2026-03-31T23:59:59Z"  # EDIT: winter window, least haze
S2_GSD=10; NPX=128  # 128 px x 10 m = 1280 m, same footprint as MEET
UAE_CLASSES=["Airport","Beach","Brownfield","Chemical plant","Construction","Highway","Landfill","Mountain","Port","Quarry","Resort","Rock","Sand","Scrub","Solar power plant","Thermal power plant"]  # resolvable at 10 m
N_UAE=15  # EDIT: max candidates per class (fewer to review)
UA="aire410-course-project (contact: kommaltariq@gmail.com)"  # EDIT: your contact for the user agent
DELAY=0.5  # EDIT: seconds between Overpass/STAC requests
ZIP_GB={"Sand":.64,"Rock":.63,"Scrub":8.97,"Dry field":55.3,"Brownfield":.71,"Construction":1.01,"Quarry":3.28,"Landfill":.66,"Mountain":1.33,"Hills":1.12,"Petroleum well":.64,"Solar power plant":21.9,"Thermal power plant":.70,"Chemical plant":.71,"Substation":.71,"Mosque":1.02,"Resort":.67,"Swimming pool":1.29,"Roundabout":.79,"Port":.75,"Airport":1.61,"Military training ground":1.21,"Highway":2.77,"Beach":.85}
MODELS=ROOT/"models"
PRACT=0.01  # EDIT: practical-significance threshold on accuracy difference
NBOOT=1000  # EDIT: bootstrap resamples