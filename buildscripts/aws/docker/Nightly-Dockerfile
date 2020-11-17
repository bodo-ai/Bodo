FROM ubuntu:20.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y wget git curl

ADD buildscripts/setup_conda.sh .
ADD buildscripts/aws/test_installs.sh .

ENV CONDA_ENV BodoCodeBuild
ENV RUN_NIGHTLY yes

RUN ./setup_conda.sh
RUN ./test_installs.sh

CMD ["/bin/bash"]
