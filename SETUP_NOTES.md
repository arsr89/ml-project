# DVC remote storage

DVC is initialized but has no remote configured yet. Point it at your storage backend:

    dvc remote add -d storage s3://your-bucket/dvc-store
    git add .dvc/config
    git commit -m "Configure DVC remote"

Then track data with:

    dvc add data/raw/your_dataset.csv
    git add data/raw/your_dataset.csv.dvc
    git commit -m "Track dataset with DVC"
    dvc push
