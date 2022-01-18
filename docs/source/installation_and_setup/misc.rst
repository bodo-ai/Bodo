.. _installation_misc:

Miscellaneous Resources
=======================

.. _passwordless_ssh:

Setting up passwordless SSH on your multi-node cluster
------------------------------------------------------

Using MPI on a multi-node cluster requires setting up passwordless SSH
between the hosts. There are multiple ways to do this. Here is one way:

1. Generate an SSH key pair using a tool like ``ssh-keygen``, for instance::

    ssh-keygen -b 2048 -f cluster_ssh_key -N ""

2. Copy over the generated private key (``cluster_ssh_key``) and public key (``cluster_ssh_key.pub``) to all the hosts and 
   store them in ``~/.ssh/id_rsa`` and ``~/.ssh/id_rsa.pub`` respectively.

3. Add the public key to ``~/.ssh/authorized_keys`` on all hosts.

4. To disable host key checking, add the following to ``~/.ssh/config`` on each host::
   
    Host *
        StrictHostKeyChecking no
