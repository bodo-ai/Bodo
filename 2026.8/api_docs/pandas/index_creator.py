# read the file, and create a list of the headings with autorefs
import re


def create_index(file_path):
    with open(file_path) as file:
        file_content = file.read()
        headings = re.findall(r"## (.+)", file_content)
        index_str = "| function | description |\n| --- | --- |\n"
        for heading in headings:
            index_str += f"| [{heading}][{heading.replace(' ', '_').replace('.', '').lower().replace('`', '')}]| |\n"

    print(index_str)


create_index("/Users/ritwika/bodo/docs/docs/api_docs/bodo_parallel_apis/index.md")
