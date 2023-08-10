# Script to build Slurm RPMs from source with the configuration required for Bodo Platform

# Currently, we run this manually when necessary (e.g. upgrading Slurm version or configs). TODO: automate
# run this script on a Linux machine compatible with our AMIs (currently Amazon Linux 2)
# docker run -it amazonlinux:2.0.20221004.0


export SLURM_VERSION="22.05.5"
curl https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2 --output slurm-${SLURM_VERSION}.tar.bz2
# Install compilers and dev deps, see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compile-software.html
yum groupinstall -y "Development Tools"
# add EPEL yum repo according to https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/add-repositories.html
yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
yum install -y tar bzip2 rpm-build munge-devel munge-libs python3 readline-devel \
    pam-devel "perl(ExtUtils::MakeMaker)" mysql-devel hwloc-devel http-parser-devel json-c-devel

# Update slurm.spec to enable auth/none and cred/none support
tar --bzip -x -f slurm-${SLURM_VERSION}.tar.bz2
sed -i 's|rm -f %{buildroot}/%{_libdir}/slurm/auth_none.so|#rm -f %{buildroot}/%{_libdir}/slurm/auth_none.so|g' slurm-${SLURM_VERSION}/slurm.spec
sed -i 's|rm -f %{buildroot}/%{_libdir}/slurm/cred_none.so|#rm -f %{buildroot}/%{_libdir}/slurm/cred_none.so|g' slurm-${SLURM_VERSION}/slurm.spec
tar -cvjSf slurm.tar.bz2 slurm-${SLURM_VERSION}

# build RPMs with slurmrestd and hwloc support (mysql is necessary to avoid a bug in Slurm build)
rpmbuild -ta slurm.tar.bz2 --with mysql --with slurmrestd --with hwloc

# install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Set AWS keys to allow upload
export AWS_ACCESS_KEY_ID="xxx"
export AWS_SECRET_ACCESS_KEY="yyy"
aws s3 cp /root/rpmbuild/RPMS/x86_64/ s3://bodo-slurm-binaries/ --recursive


## Compute nodes can install these binaries using:
# aws s3 cp s3://bodo-slurm-binaries /tmp/slurm_bin --recursive
# yum localinstall -y \
#     /tmp/slurm_bin/slurm-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-perlapi-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-devel-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-example-configs-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-slurmctld-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-slurmd-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-slurmdbd-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-libpmi-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-torque-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-openlava-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-contribs-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-pam_slurm-${SLURM_VERSION}-1.amzn2.x86_64.rpm \
#     /tmp/slurm_bin/slurm-slurmrestd-${SLURM_VERSION}-1.amzn2.x86_64.rpm
