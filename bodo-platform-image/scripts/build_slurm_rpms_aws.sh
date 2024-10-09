# Script to build Slurm RPMs from source with the configuration required for Bodo Platform

# Currently, we run this manually when necessary (e.g. upgrading Slurm version or configs). TODO: automate
# run this script on a Linux machine compatible with our AMIs (currently Rocky Linux 9)
# docker run -it -rm rockylinux:9.3

# Make sure we have a matched kernel-devel package installed
dnf install -y kernel-devel-matched kernel-headers
# Bug in rocky 9.4, updating kernel doesn't trigger rebuilding grub
grub2-mkconfig -o /boot/grub2/grub.cfg
# Reboot to make sure the kernel-devel package is loaded
reboot


export SLURM_VERSION="22.05.5"
curl https://download.schedmd.com/slurm/slurm-${SLURM_VERSION}.tar.bz2 --output slurm-${SLURM_VERSION}.tar.bz2
# Install compilers and dev deps, see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/compile-software.html
dnf groupinstall -y "Development Tools"
# add EPEL and CRB repos
dnf install -y epel-release
dnf config-manager --set-enabled crb
dnf install -y tar bzip2 rpm-build munge-devel munge-libs python3 readline-devel \
    pam-devel "perl(ExtUtils::MakeMaker)" mysql-devel hwloc-devel http-parser-devel json-c-devel dbus-devel

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
# dnf localinstall -y \
#     /tmp/slurm_bin/slurm-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-perlapi-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-devel-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-example-configs-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-slurmctld-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-slurmd-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-slurmdbd-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-libpmi-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-torque-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-openlava-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-contribs-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-pam_slurm-${SLURM_VERSION}-1.el9.x86_64.rpm \
#     /tmp/slurm_bin/slurm-slurmrestd-${SLURM_VERSION}-1.el9.x86_64.rpm
