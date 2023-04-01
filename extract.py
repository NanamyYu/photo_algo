import tarfile

with tarfile.open("data/output_generated_data.tar") as tar:
    tar.extractall()