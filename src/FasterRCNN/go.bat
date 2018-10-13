gcloud ml-engine jobs submit training JOB26 --job-dir=gs://frcnn-gpu --module-name=FasterRCNN.train_frcnn --package-path=FasterRCNN --region=us-central1 --config=FasterRCNN\cloudml-gpu.yaml -- -o rsna -p gs://frcnn-gpu

gcloud ml-engine jobs submit training JOB26 --job-dir=gs://frcnn-gpu --module-name=FasterRCNN.train_frcnn --package-path=FasterRCNN --region=us-central1 --config=FasterRCNN\cloudml-gpu.yaml -- -o rsna -p gs://frcnn-gpu --prev-weights=prev_model.hdf5

gcloud ml-engine jobs cancel JOB28