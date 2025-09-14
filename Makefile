.PHONY: setup normalize-pheme train-pheme-lr train-pheme-svm eval-pheme-lr app

setup:
\tpip install -r requirements.txt

normalize-pheme:
\tpython -m rumor_detection.normalize_pheme_csv --in data/pheme/dataset.csv --out data/pheme/pheme_norm.csv

train-pheme-lr:
\tpython -m rumor_detection.train --model lr  --data data/pheme/pheme_norm.csv --text-col text --label-col label --save-dir models

train-pheme-svm:
\tpython -m rumor_detection.train --model svm --data data/pheme/pheme_norm.csv --text-col text --label-col label --save-dir models

eval-pheme-lr:
\tpython -m rumor_detection.evaluate --data data/pheme/pheme_norm.csv --text-col text --label-col label --model-path models/lr_tfidf.joblib --report-path reports/metrics_eval.json --confusion-path reports/img/confusion_pheme.png

app:
\tstreamlit run app/streamlit_app.py
