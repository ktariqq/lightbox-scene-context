from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA,SUBSET,FEAT,RES,UAE,UAE_SUBSET=[ROOT/p for p in ("data","subset","features","results","uae","uae_subset")]
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
RES_MAX=1.5  # EDIT: flag UAE source resolution above this (m)
BLUR_MIN=20.0  # EDIT: flag UAE Laplacian variance below this
UA="aire410-course-project (contact: kommaltariq@gmail.com)"  # EDIT: your contact for the user agent
DELAY=0.5  # EDIT: seconds between Esri/Overpass requests
ZIP_GB={"Sand":.64,"Rock":.63,"Scrub":8.97,"Dry field":55.3,"Brownfield":.71,"Construction":1.01,"Quarry":3.28,"Landfill":.66,"Mountain":1.33,"Hills":1.12,"Petroleum well":.64,"Solar power plant":21.9,"Thermal power plant":.70,"Chemical plant":.71,"Substation":.71,"Mosque":1.02,"Resort":.67,"Swimming pool":1.29,"Roundabout":.79,"Port":.75,"Airport":1.61,"Military training ground":1.21,"Highway":2.77,"Beach":.85}
MODELS=ROOT/"models" 
PRACT=0.01  # EDIT: practical-significance threshold on accuracy difference
NBOOT=1000  # EDIT: bootstrap resamples
