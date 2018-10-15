python3 jpg2tfrecord.py -d gs://nih-chest-xrays/data  -i gs://nih-chest-xrays/data/un  -t gs://nih-chest-xrays/data/tfrecords &&
gsutil -m cp -r gs://nih-chest-xrays/data/tfrecords  gs://kaggle-rsna-datastorage/nih-chest-xrays/tfrecords
