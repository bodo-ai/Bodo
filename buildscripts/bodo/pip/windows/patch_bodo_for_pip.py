import argparse
import os
import shutil
import zipfile


def patch_init(wheel_dir: str) -> None:
    with open(os.path.join(wheel_dir, "bodo", "__init__.py")) as init_f:
        py_init_text: str = init_f.read()

    contents = py_init_text.split('"""', 2)
    header, code = contents[1], contents[2]

    script_dir = os.path.dirname(__file__)

    with open(os.path.join(script_dir, "add_dlls.py")) as patch_f:
        patch = patch_f.read()

    patched_init_text = f'"""{header}"""\n\n{patch}{code}'

    with open(os.path.join(wheel_dir, "bodo", "__init__.py"), "w") as init_f:
        init_f.write(patched_init_text)


def patch_wheels(path: str) -> None:
    for wheel in os.listdir(path):
        print("\nPATCHING", wheel)

        with zipfile.ZipFile(os.path.join(path, wheel), "r") as z:
            z.extractall("whl_tmp")

        patch_init("whl_tmp")

        # This packs the wheel back
        with zipfile.ZipFile(os.path.join(path, wheel), "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk("whl_tmp"):
                for f in files:
                    z.write(
                        os.path.join(root, f),
                        os.path.relpath(os.path.join(root, f), "whl_tmp"),
                    )
        shutil.rmtree("whl_tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        help="Path to a wheel or directory containing .so files to patch",
    )
    args = parser.parse_args()
    patch_wheels(args.path)
