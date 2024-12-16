# To Update:
# docker build -t cibuildwheel-manylinux:latest -f buildscripts/bodo/pip/manylinux/Dockerfile --platform linux/amd64 .
# docker tag cibuildwheel-manylinux:latest 427443013497.dkr.ecr.us-east-2.amazonaws.com/cibuildwheel-manylinux:latest
# docker push 427443013497.dkr.ecr.us-east-2.amazonaws.com/cibuildwheel-manylinux:latest 
FROM mambaorg/micromamba:1.5.8 AS micromamba

FROM quay.io/pypa/manylinux_2_35_x86_64:latest

USER root

ARG MAMBA_USER=root
ARG MAMBA_USER_ID=0
ARG MAMBA_USER_GID=0
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN  /usr/local/bin/_dockerfile_initialize_user_accounts.sh && /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

CMD ["/bin/bash"]
