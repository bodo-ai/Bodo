FROM ubuntu:20.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y wget git

ADD buildscripts/bodosql/install.sh .

ENV CONDA_ENV BodoSQL

RUN ./install.sh

CMD ["/bin/bash"]
