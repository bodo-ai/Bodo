import os
from tempfile import TemporaryDirectory


def test_build():
    print(
        "=============================\nRunning Test Build\n=============================\n"
    )

    with TemporaryDirectory() as temp_dir:
        src_dir = os.path.join(os.path.dirname(__file__), "test-build")
        output_dir = os.path.join(temp_dir, "build")

        build_output = os.system(
            "sphinx-build -b html {} {}".format(src_dir, output_dir)
        )

    assert build_output == 0

    print(
        "\n=============================\nTest Build Complete\n=============================\n"
    )
