#!/usr/bin/env python3
import argparse
import re
import sys

PCT_RE = re.compile(r"(\d+)%\s*$")
BEFORE_WORD_RE = re.compile(r"\bbefore\b\s+(\S+)\s+(\S+)", re.IGNORECASE)


def last_percent(line):
    m = PCT_RE.search(line)
    return int(m.group(1)) if m else None


def word_after_before(line):
    m = BEFORE_WORD_RE.search(line)
    return (m.group(1), m.group(2)) if m else (None, None)


def process_lines(lines):
    results = []
    for i in range(len(lines) - 1):
        a = lines[i].rstrip("\n")
        b = lines[i + 1].rstrip("\n")
        if "before" in a.lower() and "after" in b.lower():
            w0, w1 = word_after_before(a)
            pa = last_percent(a)
            pb = last_percent(b)
            if (
                w0 is not None
                and w1 is not None
                and pa is not None
                and pb is not None
                and pa != pb
            ):
                results.append((w0, w1, pa, pb, a, b))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find before/after line pairs with differing final percentages."
    )
    parser.add_argument(
        "file", nargs="?", help="Path to log file. If omitted, read stdin."
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    matches = process_lines(lines)
    # Print header
    if matches:
        print("word_after_before\tbefore_pct\tafter_pct")
        for w0, w1, pa, pb, a, b in matches:
            print(f"{w0}\t{w1}\t{pa}%\t{pb}%")
    else:
        print(
            "No matching before/after pairs with differing final percentages found.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
