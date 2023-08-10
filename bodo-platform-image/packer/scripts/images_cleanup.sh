#!/bin/bash

# Make sure Udev doesn't block our network
echo "cleaning up udev rules"
rm -rf /etc/udev/rules.d/70-persistent-net.rules

echo "cleaning up eth0 HWADDR and UUID"
if [ -r /etc/sysconfig/network-scripts/ifcfg-eth0 ]; then
  sed -i 's/^HWADDR.*$//' /etc/sysconfig/network-scripts/ifcfg-eth0
  sed -i 's/^UUID.*$//' /etc/sysconfig/network-scripts/ifcfg-eth0
fi

echo "--- Force logrotate"
/usr/sbin/logrotate -f /etc/logrotate.conf
/bin/rm -f /var/log/*-???????? /var/log/*.gz