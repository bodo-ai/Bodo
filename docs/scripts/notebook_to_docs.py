"""
Script for automatically converting tutorial notebooks into docs pages.

Usage:
    python docs/scripts/notebook_to_docs.py path/to/notebook.ipynb --output-dir docs/docs/guides --tag tag
"""

import pathlib
import subprocess
from argparse import ArgumentParser

GITHUB_REPO = "https://github.com/bodo-ai/Bodo"
BRANCH = "main"


def main():
    parser = ArgumentParser(
        description="Convert Jupyter notebooks to markdown for docs."
    )
    parser.add_argument(
        "notebook", type=pathlib.Path, help="Path to the Jupyter notebook file."
    )
    parser.add_argument(
        "--outdir", type=pathlib.Path, help="Output directory of the markdown file."
    )
    parser.add_argument(
        "--tag", type=str, help="tag to reference page in other parts of the site."
    )
    args = parser.parse_args()

    nb = args.notebook
    out_dir = args.outdir
    tag = args.tag

    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            "--output-dir",
            str(out_dir),
            str(nb),
        ],
        check=True,
    )

    md_file = out_dir / f"{nb.stem}.md"

    # prepend the link, tag and title to the markdown
    text = md_file.read_text(encoding="utf-8")
    splittext = text.split("# ", 1)
    splittext = splittext[1].split("\n", 1)
    title = splittext[0]
    nb_path = nb.as_posix()

    link = f"[View Notebook on GitHub]({GITHUB_REPO}/blob/{BRANCH}/{nb_path})"

    comment = f"""
<!-- 
To make changes to this page, first make your changes in the notebook: {nb_path}
then run the script from the project directory: python docs/scripts/notebook_to_docs.py {nb_path} --outdir {out_dir.as_posix()} --tag {tag}
-->
"""

    tagtext = "{#" + tag + "}"
    page_header = f"{title} {tagtext}\n================="

    md_file.write_text(
        f"{comment}\n\n{page_header}\n\n{link}\n\n{splittext[1]}", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
