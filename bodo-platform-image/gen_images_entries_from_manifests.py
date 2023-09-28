import json
import os
import re

AMI_BUILDER, VMI_BUILDER = (
    "amazon-ebs",
    "azure-arm",
)  # builder type for AMIs and VMIs in the packer template


def gen_manifest_file_prefix_map(type_list):
    return {t: f"{t}_manifest_" for t in type_list}


def gen_manifest_fpaths_map(file_prefix_map, manifest_files):
    return {
        t: list(sorted(filter(lambda x: x.startswith(t_prefix), manifest_files)))
        for t, t_prefix in file_prefix_map.items()
    }


def get_builder_type(fpath, idx):
    """
    Given fpath to a packer manifest file,
    return the builder type of the (idx)th build
    """
    with open(fpath, "r") as f:
        manifest = json.load(f)

    assert "builds" in manifest, "packer build did not build anything"
    assert (
        len(manifest["builds"]) > idx
    ), f"one of the builds failed, successful ones are {list_builder_types(fpath)}"
    assert "builder_type" in manifest["builds"][idx], "no builder_type found"
    return manifest["builds"][idx]["builder_type"]


def list_builder_types(fpath):
    """
    Given fpath to a packer manifest file,
    return the list of builder types
    """
    with open(fpath, "r") as f:
        manifest = json.load(f)

    assert "builds" in manifest, "packer build did not build anything"
    builds_size = len(manifest["builds"])
    res = []
    for i in range(builds_size):
        res.append(get_builder_type(fpath, i))
    return res


def get_builder_type_idx(manifest, builder_type):
    idx = [
        manifest["builds"][i]["builder_type"] for i in range(len(manifest["builds"]))
    ].index(builder_type)
    return idx


def get_custom_data_value(fpath, key, builder_type):
    """
    Given fpath to a packer manifest file,
    return the value for key from custom_data the of the (idx)th build
    """
    with open(fpath, "r") as f:
        manifest = json.load(f)

    assert "builds" in manifest, "packer build did not build anything"
    assert builder_type in list_builder_types(
        fpath
    ), f"Build for {builder_type} not found in {fpath}"
    idx = get_builder_type_idx(manifest, builder_type)
    assert "custom_data" in manifest["builds"][idx], "no custom data found"
    assert key in manifest["builds"][idx]["custom_data"], "custom data is empty"

    return manifest["builds"][idx]["custom_data"][key]


def get_img_prefix(fpath, builder_type):
    """
    Given fpath to a packer manifest file,
    return the img_prefix from custom_data
    """

    return get_custom_data_value(fpath, "img_prefix", builder_type)


def get_bodo_version(fpath, builder_type):
    """
    Given fpath to a packer manifest file,
    return the bodo_version from custom_data
    """
    return get_custom_data_value(fpath, "bodo_version", builder_type)


def get_vmi_id(fpath):
    """
    Given fpath to a packer manifest file
    the shared image id
    """
    with open(fpath, "r") as f:
        manifest = json.load(f)

    assert "builds" in manifest
    assert len(manifest["builds"]) > 0
    idx = get_builder_type_idx(manifest, VMI_BUILDER)
    assert "artifact_id" in manifest["builds"][idx]
    assert manifest["builds"][idx]["builder_type"] == "azure-arm"
    img_id = manifest["builds"][idx]["artifact_id"]
    match = re.search("(.*)Microsoft.Compute(.*)", img_id)
    shared_img_id = (
        match.group(1) + "Microsoft.Compute/galleries/BodoImages" + match.group(2)
    )

    return shared_img_id


def get_ami_ids_by_region(fpath):
    """
    Given fpath to a packer manifest file,
    return a dict that maps region to ami_id
    """

    with open(fpath, "r") as f:
        manifest = json.load(f)

    assert "builds" in manifest
    assert len(manifest["builds"]) > 0
    idx = get_builder_type_idx(manifest, AMI_BUILDER)
    assert "artifact_id" in manifest["builds"][idx]
    assert manifest["builds"][idx]["builder_type"] == "amazon-ebs"

    ami_ids_str = manifest["builds"][idx]["artifact_id"]
    ami_ids_region_split = ami_ids_str.split(",")
    ami_id_by_region = dict(
        [(x.split(":")[0], x.split(":")[1]) for x in ami_ids_region_split]
    )
    return ami_id_by_region


IMG_TYPES = ["worker"]

manifest_files = list(filter(lambda x: x.endswith(".json"), os.listdir(".")))
img_type_manifest_file_prefix_map = gen_manifest_file_prefix_map(IMG_TYPES)

# Mapping from image type to list of manifest files for that image type
img_type_manifest_fpaths_map = gen_manifest_fpaths_map(
    img_type_manifest_file_prefix_map, manifest_files
)

NUMBER_OF_BODO_VERSIONS = len(img_type_manifest_fpaths_map[IMG_TYPES[0]])
# Hardcode azure regions because shared image replication regions is not stored in manifest file
# This list must be consistent with replication regions specified in images.json
AZURE_REGIONS = [
    "eastus",
    "eastus2",
    "westus",
    "westus2",
    "westus3",
    "centralus",
    "northcentralus",
    "southcentralus",
    "westcentralus",
]
assert "GIT_SHA" in os.environ
GIT_SHA = os.environ["GIT_SHA"]

# Make sure each image type has same number of manifest files
# assert (
#     len(
#         set(
#             len(t_manifest_files)
#             for t_manifest_files in img_type_manifest_fpaths_map.values()
#         )
#     )
#     == 1
# ), "Some Bodo versions do not have manifest files"

# CI packer build step can be green even though packer build failed

for image_type, manifest_fpaths_map in img_type_manifest_fpaths_map.items():
    for i in range(NUMBER_OF_BODO_VERSIONS):
        builder_types = list_builder_types(manifest_fpaths_map[i])
        # Make sure each AMI type manifest has AMI successfully generated
        assert (
            AMI_BUILDER in builder_types
        ), f"AMI build for {manifest_fpaths_map[i]} failed. Successful builds: {builder_types}"
        #         # Make sure each VM type manifest has VMI successfully generated
        assert (
            VMI_BUILDER in builder_types
        ), f"VMI build for {manifest_fpaths_map[i]} failed. Successful builds: {builder_types}"

# TODO assert that the indices are the same in all.

# Creating image create requests
img_create_requests = []
for idx in range(NUMBER_OF_BODO_VERSIONS):
    ami_types_manifest_fpath_map = {
        ami_type: manifest_fpaths_map[idx]
        for ami_type, manifest_fpaths_map in img_type_manifest_fpaths_map.items()
    }

    vmi_types_manifest_fpath_map = {
        vmi_type: manifest_fpaths_map[idx]
        for vmi_type, manifest_fpaths_map in img_type_manifest_fpaths_map.items()
    }

    img_prefix = get_img_prefix(ami_types_manifest_fpath_map[IMG_TYPES[0]], AMI_BUILDER)
    bodo_version = get_bodo_version(
        ami_types_manifest_fpath_map[IMG_TYPES[0]], AMI_BUILDER
    )

    # This will end up being of the form
    # {'worker': {'us-east-1': 'ami-XXXX', 'us-east-2': 'ami-XXXX', ...}}
    ami_ids_by_type_by_region = {
        t: get_ami_ids_by_region(ami_types_manifest_fpath_map[t])
        for t in ami_types_manifest_fpath_map.keys()
    }

    vmi_ids_by_type = {
        t: get_vmi_id(vmi_types_manifest_fpath_map[t])
        for t in vmi_types_manifest_fpath_map.keys()
    }

    # Only the regions that which have all types of AMIs. Guarantees
    # complete entries in the bodo_ami table in backend
    regions_intersection = set.intersection(
        *[set(t.keys()) for _, t in ami_ids_by_type_by_region.items()]
    )

    # Basically a inverse of ami_ids_by_type_by_region
    ami_ids_by_region_by_type = {
        region: {
            ami_type: ami_ids_by_region[region]
            for ami_type, ami_ids_by_region in ami_ids_by_type_by_region.items()
        }
        for region in regions_intersection
    }

    vmi_ids_by_region_by_type = {
        region: {vmi_type: vmi_ids_by_type[vmi_type] for vmi_type in IMG_TYPES}
        for region in AZURE_REGIONS
    }

    ami_create_requests = [
        {
            "bodoVersion": bodo_version,
            "supported": True,
            "imageGitSha": GIT_SHA,
            "imagePrefix": img_prefix,
            "region": region,
            "workerImageId": ami_ids_by_type["worker"],
            "imageCloudProvider": "AWS",
        }
        for region, ami_ids_by_type in ami_ids_by_region_by_type.items()
    ]
    img_create_requests.extend(ami_create_requests)

    vmi_create_requests = [
        {
            "bodoVersion": bodo_version,
            "supported": True,
            "imageGitSha": GIT_SHA,
            "imagePrefix": img_prefix,
            "region": region,
            "workerImageId": vmi_ids_by_type["worker"],
            "imageCloudProvider": "AZURE",
        }
        for region, vmi_ids_by_type in vmi_ids_by_region_by_type.items()
    ]
    img_create_requests.extend(vmi_create_requests)

with open("img_share_requests.json", "w") as f:
    json.dump(img_create_requests, f)
