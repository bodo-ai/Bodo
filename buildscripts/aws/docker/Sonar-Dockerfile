FROM ubuntu:20.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y wget git python3 python3-pip unzip

ADD buildscripts/aws/sonar_installs.sh .

ENV CONDA_ENV BodoCodeBuild

RUN ./sonar_installs.sh
CMD ["/bin/bash"]
