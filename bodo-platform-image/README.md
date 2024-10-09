# bodo-images

This folder contains scripts for:

1. Building Images(AMI&VM Image)'s containing Bodo executables
2. Updating the list of available AMIs to amazon account ID's of users allowed
3. Updating the list of available VMIs

## Overview

A machine image is a single static unit that contains a pre-configured operating system and installed software which is used to quickly create new running machines.
An AMI (Amazon Machine Image) is one format example of machine images. It is used specifically for creating EC2 instances.
Similarly, VM Image is used specifically for creating Azure VM instances.

In our case, machines images are useful for supporting the Bodo platform for running Clusters on AWS/AZURE.
Specifically, we create images for the worker nodes in a given cluster and for nodes hosting the Notebook itself.
We would like cluster nodes to all have Bodo installed on them with the right configuration.

### Tools for automating the Image creation

#### Use Packer for automating the process of Image creation

Images creation is done using the [Packer tool](https://www.packer.io/). For those familiar with Terraform, one might ask, is Packer and terraform the same thing? Not quiet. Terraform specifies the desired configuration for creating AWS/AZURE instances. When time comes, it creates the instance from scratch using that configuration. On the other hand, packer bakes a machine image (AMI/VM Image) putting everything in it, which makes launching instance based on the AMI/VM Image faster than Terraform.

Packer works in 3 steps: (1) launch a source Image(AMI/VM Image) on a machine(EC2/VM), (2) provision it, and (3) repackage it into a new image.
Packer automatically cleans up resources created during creating images, including the launched source image.
The provisioning part is where we setup the configuration of the image. Here, Packer would get the help of a configuration manager to do the job (we use Ansible in this case).

#### Use Ansible for configuration management

Ansible automates the process of configuring hosts with different configurations. It performs the bulk of the provisioning task for Packer, from installing dependencies, to system configuration, copying files to the host machine, executing setup scripts, etc.

## Folder Structure

#### gen_images_entries_from_manifests.py

Generates `ami_share_requests.json` and `vmi_share_requests.json` from manifest files produced from `packer build`.

#### stage_azure_img_def.sh

Stage Azure image definition in the Azure shared image gallery.

### packer/ directory

#### templates/

Templates are the HCL configuration files used by packer to describe what images we want to build and how.
`images.pkr.hcl` is used to deploy new images.

In the `images.pkr.hcl` file, you will find 3 main sections: variables, builders, and provisioners. There's a 4th post-processors section which generates the manifest files from the packer build.

Variables is the map of variable names to values. They can be overriden on the cmd line if needed. The `{{env ...}}` mechanism lets you read environment variable values in.

Builders do the building of the machine, so you find machine creation specific details there. We have 2 builders: `amazon-ebs.build` and `azure-arm.build`.

Provisioners take care of configuration management. We run a `ami_precondition.sh`/`vm_precondition.sh` script to install `ansible`, our configuration manager. Then we run the ansible playbook for the specific images we are building. At the end, we perform some cleanups. Provisioning steps vary between AMI and Azure Images.

#### scripts/

This directory has the bash scripts that we call from the template for setup and cleanup before and after the ansible run respectively.

#### vars/

This directory has user defined variables for the different configurations we run, to be passed at runtime using the `var-file` flag. Look at the [build section](#local-build) for details on how it is used.

### ansible/ directory

A more comprehensive guide to the directory structure used by ansible please refer to [this](https://docs.ansible.com/ansible/latest/user_guide/playbooks_reuse_roles.html#id5) guide.

#### playbooks

The YAML files under the `ansible/` directory are called playbooks. A playbook is a map from hosts (machines to deploy the configs on) to roles. Roles are lists of tasks to be performed.

We have 2 different playbooks, one for each node type we would like to create images for.

- `worker_playbook.yaml` is for cluster nodes that will carry the work. It will mainly have bodo and mpi.

We use Intel-MPI by default on both AWS and Azure. On AWS we also install the EFA drivers. Note that it's fine to have the EFA drivers installed even when the image
is used on a non-EFA enabled instance.

Since we do not perform more than one machine deployment at a time, we have one playbook for each deploy type and in each of the playbooks we find only one play (a play is a host to roles map)

TODO: merge all playbooks as multiple plays in on playbook???

#### roles/ directory

Each role will have a directory containing at least one directory of the tasks to be performed by the role.

##### roles/x/defaults/ directory

If roles/x/defaults/main.yml exists, variables listed therein will be added to the play.
It will have the default values for variables to be used if not specified otherwise.

##### roles/x/files/ directory

Files and scripts that can be used by role x during playbook execution.

##### roles/x/tasks/ directory

The core of the role is the tasks it executes. This is where they are specified as YAML files.

##### roles/x/templates/ directory

This directory contains Jinja2 (.j2 extension) templates of files we would want to port to the server we are configuring. Template files would have variables with dynamic values set at the time of playbook execution.

#### vars/ directory

This directory contains the definitions of vars for the different roles.

## Image rebuilding

Image rebuilding is triggered automatically from the Bodo engine repo whenever a release happens. You can manually rebuild and share Images for a new Bodo version manually by (1) manually uploading the new binary to the Artifactory channel `bodo.ai-platform` and then (2) trigger the pipeline through the UI.

## Image sharing

Image sharing is done automatically.

- AWS: The pipeline contacts the different environments (dev, staging, and prod) and shares the details about newly created AMIs with them. The platform then shares these images with customer accounts. AMI IDs are the identifications for AMIs. Each region has a unique AMI ID.

- AZURE: The pipeline [manually stage](https://github.com/Bodo-inc/bodo-ami/blob/master/stage_azure_img_def.sh)(due to limitations of packer) image definitions in the shared image gallery, then create VM images and make the created image version 1.0.0 for the image definition. The pipeline contact the different environments and shares the details about newly created VMIs with them. Images versions of the image definitions are the identification for VMIs. Image definitons are replicated in all US regions and all regions have the same VMI ID.
  - We use shared image gallery to make our images available to the customer accounts. An app registration in the Azure Bodo account has permission to the _Bodo Image_ shared image gallery. When we have a new customer, we give them contributor access to our app registration, and they then have acess to images in the shared image gallery. For more information, see [this](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/share-images-across-tenants).

## Image Testing

To test the images with our platform when making changes to the images, do the following:

- Run pipeline from your branch manually as PR pipeline will not create any images.
- Go to the CI(Github actions) for the specific pipeline.
- Download and view the created artifact(s) (`*-worker-manifest`).
- Find the image ID(s) for new image(s) in the manifest artifacts.
- Add new entry to the bodo_images table in the [dev database](https://pgadmin-dev.bodo.ai/), and set `supported` to true. If there are other images with the same platform, region and version, set `supported` to false for the old image. The image region needs to be the same as the account settings region.
- The image is now available to the dev platform for testing.

## Source Image OS

- AMI: RockyLinux 9 x86_64 
- VMI: AlmaLinux 9 x86_64 

## Local build

In order to build new Images(AMI&VM Image using the same template, specified by different builders) from the latest Bodo release (we keep our releases in an internal Artifactory channel) and push them to region specified in `templates/images.pkr.hcl` use:

```
cd packer
packer build -var 'node_role=worker' -var 'playbook_file=worker_playbook.yaml' templates/images.pkr.hcl
```

or you can use json variables files stored in vars directory:

```
cd packer
packer build -var-file vars/worker.json templates/images.pkr.hcl
```

## Confluence Docs

- Testing workflow: https://bodo.atlassian.net/wiki/spaces/BP/pages/1079967884/Bodo+Imaging+101
- Bodo release: https://bodo.atlassian.net/wiki/spaces/BP/pages/1139703813/Bodo+Image+Release
- Updating Intel MPI: https://bodo.atlassian.net/wiki/spaces/BP/pages/865861645/Upgrading+Intel+MPI
- Testing plan: https://bodo.atlassian.net/wiki/spaces/BP/pages/1108115457/AMI+testing
