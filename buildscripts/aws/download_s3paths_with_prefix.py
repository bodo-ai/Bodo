"""
    Script to run on AWS Codebuild. It takes in a bucket and AWS prefix
    and downloads all of the directories and files referenced, maintaining
    the directory structure
"""

# Copied mostly from https://stackoverflow.com/questions/49772151/download-a-folder-from-s3-using-boto3
import argparse
import boto3
import os


def download_contents(bucketname, prefix):
    # Account configuration should be handled by Codebuild
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucketname)
    for obj in bucket.objects.filter(Prefix=prefix):
        target = obj.key
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        bucket.download_file(obj.key, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bucketname", type=str, help="Name of the bucket.")
    parser.add_argument("prefix", type=str, help="Prefix used to select contents")
    args = parser.parse_args()
    download_contents(args.bucketname, args.prefix)
