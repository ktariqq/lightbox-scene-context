set -e
case "$1" in
  prep) python 01_manifest.py ;;  # then EDIT: fill the arid list in classes.json and push the repo to Kaggle/Colab
  remote) python 02_extract.py --workers 16; python 03_audit.py a; python features.py train val test ;;  # run on Kaggle or Colab, zip and download subset/ features/ manifests
  audit) python 03_audit.py a; python 03_audit.py b ;;
  uaetest) python 04_uae_candidates.py; python 05_uae_fetch.py test ;;  # inspect the printed assets and scene before continuing
  uaefetch) python 05_uae_fetch.py run; python 06_uae_review.py --out review_s2.html ;;  # then EDIT: review uae/review_s2.html manually and save uae/labels.csv
  uaeviews) python 07_uae_views.py; python features.py uae ;;
  meet10) python 08_meet10.py; python features.py meet10 ;;
  train) python train.py b1; python train.py tune; python train.py heads ;;
  train-fast) python train.py b1; python train.py heads default ;;  # fallback: no tuning, one default config
  report) python evaluate.py; python 09_s2_summary.py; python explain.py all; python export.py b4vd 42 ;;
  freeze) pip freeze > requirements.lock.txt ;;
  *) echo "stages: prep remote audit uaetest uaefetch uaeviews meet10 train|train-fast report freeze"; exit 1 ;;
esac