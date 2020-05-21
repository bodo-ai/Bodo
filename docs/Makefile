# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Bodo
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile source/getting_started.rst source/user_guide.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

source/getting_started.rst:
	if [ ! -d "Bodo-tutorial" ]; then \
		git clone https://github.com/Bodo-inc/Bodo-tutorial.git; \
	fi
	jupyter nbconvert --to rst Bodo-tutorial/bodo_getting_started.ipynb
	python convert-tutorial.py Bodo-tutorial/bodo_getting_started.rst source/getting_started.rst
	rm Bodo-tutorial/bodo_getting_started.rst

source/user_guide.rst:
	if [ ! -d "Bodo-tutorial" ]; then \
		git clone https://github.com/Bodo-inc/Bodo-tutorial.git; \
	fi
	jupyter nbconvert --to rst Bodo-tutorial/bodo_tutorial.ipynb
	python convert-tutorial.py Bodo-tutorial/bodo_tutorial.rst source/user_guide.rst
	rm Bodo-tutorial/bodo_tutorial.rst
	cp -r Bodo-tutorial/bodo_tutorial_files Bodo-tutorial/img source

clean:
	rm -rf Bodo-tutorial source/bodo_tutorial_files source/img
	rm -f source/getting_started.rst source/user_guide.rst
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
