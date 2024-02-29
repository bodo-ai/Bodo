# parse all level 4 headings in a markdown file, and create a file with the same name as the heading, and write the content of the heading to the file

import os
import re


def create_files(file_path, output_dir):
    with open(file_path, 'r') as file:
        file_content = file.read()
        headings = re.findall(r'## (.+)', file_content)

        for heading in headings:
            file_name = heading.replace(' ', '_').replace("`", "").lower().replace('bodo.',"") + '.md'
            heading_content = re.search(r'## ' + heading + r'(.+?)(?=##|$)', file_content, re.DOTALL)
            heading_content = heading_content.group(1)
            heading_content = heading_content
            with open(os.path.join(output_dir, file_name), 'w') as output_file:
                heading = "# " + heading
                output_file.write(heading)
                heading_content = heading_content.replace("- `pandas", "`pandas")
                heading_content = heading_content.replace("***Example Usage***", "### Example Usage")
                heading_content = heading_content.replace("***Supported Arguments***", "### Supported Arguments")
                output_file.write(heading_content)


create_files('/Users/ritwika/bodo/docs/docs/api_docs/bodo_parallel_apis/bodo_parallel_apis.md',
             '/Users/ritwika/bodo/docs/docs/api_docs/bodo_parallel_apis')
