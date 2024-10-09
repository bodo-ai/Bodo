import argparse
import gzip
import re
import shutil
import sys

import boto3

PYTEST_NAME_RE = re.compile("[A-Za-z0-9-_\.]+::[A-Za-z0-9-_]+(\[[A-Za-z0-9-_]+\])?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "pull_build_logs",
        description="Given a build batch ID, pulls logs for all builds",
        usage="pull_build_logs.py [Bodo-PR-Testing:5a8a9b74-38b9-4f93-bfd6-0700e6edac9e]",
    )
    parser.add_argument("build_batch_id", nargs="*", type=str)
    parser.add_argument(
        "--pytest-failures",
        action="store_true",
        default=False,
        help="Extract pytest failures and print a list of failing tests. Disables output",
    )
    parser.add_argument(
        "--force-output",
        action="store_true",
        default=False,
        help="The default behavior is to dump all logs to stdout. However, if any of the analysis options are given, this is disabled",
    )

    args = parser.parse_args()

    do_output = True
    if not args.force_output:
        if args.pytest_failures:
            do_output = False

    codebuild = boto3.client("codebuild")
    s3 = boto3.client("s3")
    batches = codebuild.batch_get_build_batches(ids=args.build_batch_id)
    for notFound in batches["buildBatchesNotFound"]:
        print(f"WARNING: Batch {notFound} could not be found", file=sys.stderr)
    for batch in batches["buildBatches"]:
        if batch["logConfig"]["s3Logs"]["status"] != "ENABLED":
            print("WARNING: S3 Logs not enabled for batch", file=sys.stderr)
            continue

        s3_bucket, s3_loc = batch["logConfig"]["s3Logs"]["location"].split("/", 1)
        for build in batch["buildGroups"]:
            log_id = build["currentBuildSummary"]["arn"].split(":")[-1]
            s3_key = f"{s3_loc}/{log_id}.gz"

            try:
                result = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            except Exception:
                print("Logs could not be found for ", build, file=sys.stderr)
                continue

            with result["Body"] as compressed_body:
                with gzip.open(compressed_body, "rt") as body:
                    if args.pytest_failures:
                        seen = set()
                        for line in body:
                            if do_output:
                                sys.stdout.write(line)
                            if "FAILED" in line and (m := PYTEST_NAME_RE.search(line)):
                                test = m.group(0)
                                if test in seen:
                                    continue
                                seen.add(test)
                                print(test)
                    else:
                        shutil.copyfileobj(body, sys.stdout)
