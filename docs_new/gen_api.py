import markdown
import glob
from bodoapiextension import bodoapi
import re

sourcedir = 'api_docs'
targetdir = "docs/"
i = 0
for filename in glob.iglob(sourcedir + '**/**', recursive=True):

    if filename.endswith(".md"):
        i+=1
        print(i)
        with open(filename, 'r') as f:
            new_mk = ""
            text = f.readlines();
            for line in text:
                if ("++") in line:
                    line = markdown.markdown(line, extensions=[bodoapi()])
                    if "<p>" in line:
                        line = line.replace("</p>", "</code><br>").replace("<p>","<code>")
                    if "<li>" in line:
                        line = line.replace("</li>", "</code>").replace("<li>","- <code>")
                    if "<ul>" in line:
                        line = line.replace("<ul>","").replace("</ul>","")
                    line+= "<br><br>"

                new_mk += line
        outfilename = targetdir + filename
        o = open(outfilename,'w')
        o.write(new_mk)
        o.close()