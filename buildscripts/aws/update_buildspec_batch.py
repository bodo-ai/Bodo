"""
    Script used for automatically updating the batch portion of the
    buildspec.yml file. This can be used either when the file need
    to be updated (e.g. to increase the number of builds running
    concurrently), or a new buildspec file needs to be constructed
    (e.g. creating a new project for nightly tests).
"""

import argparse
from itertools import product

import yaml

CI_BUILDSPEC_FILENAME = "buildspec.yml"

# Helper function for creating cross product of the provided values. From:
# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def dict_product(dict_to_merge):
    return list(
        dict(zip(dict_to_merge.keys(), values))
        for values in product(*dict_to_merge.values())
    )


def generate_np2_vars(env_vars):
    """Helper function that creates the env vars for NP=2 tests."""
    output_dict = dict()
    for key, value in env_vars.items():
        if key == "NP":
            output_dict[key] = [2]
        elif key == "PYTEST_MARKER":
            output_dict[key] = ["smoke"]
        elif key == "NUMBER_GROUPS_SPLIT":
            output_dict[key] = [1]
        else:
            output_dict[key] = value
    return dict_product(output_dict)


# TODO(Nick): Change the batch construction to produce a graph
# and use the build as a shared step across builds with the same image.
# Inputs:
# env_vars: List of dictionarys mapping env variables -> list(values)
# images: dict[imagename used for batch naming] -> image path for each docker image.
# Outputs:
# A dictionary describing the batch portion of the yaml file
# containing the cross product of the env_vars and images.
def construct_batch_field(env_vars, images):
    build_graph = []
    merged_env_vars = dict_product(env_vars)
    # Add np=2 separately because we only want one build
    merged_env_vars.extend(generate_np2_vars(env_vars))
    for env_var_group in merged_env_vars:
        for image_name, image_path in images.items():
            buildspec = get_buildspec_file(env_var_group, image_path)
            compute_type = get_compute_type(env_var_group, image_path, buildspec)
            # Produce a unique identifier for each build instance.
            # ImageName + _ + var1key + _ + var1value + ...
            vars_list = []
            for key, value in env_var_group.items():
                vars_list.append(
                    str(key).replace(" ", "_") + "_" + str(value).replace(" ", "_")
                )
            vars_list.sort()
            identifier = image_name.replace(" ", "_") + "_" + "_".join(vars_list)
            # Now create the env dict portion
            env_dict = dict()
            env_dict["compute-type"] = compute_type
            env_dict["image"] = image_path
            env_dict["variables"] = env_var_group
            build_graph.append(
                {"identifier": identifier, "env": env_dict, "buildspec": buildspec}
            )
    # IF NP=1 is in the env vars add SonarQube as a last step
    if "NP" in env_vars and 1 in env_vars["NP"]:
        add_sonar(build_graph)
    return {"build-graph": build_graph}


def add_sonar(build_graph):
    dependencies = []
    for build in build_graph:
        if "NP" in build["env"]["variables"] and build["env"]["variables"]["NP"] == 1:
            dependencies.append(build["identifier"])
    buildspec = "buildscripts/aws/buildspecs/sonar_buildspec.yml"
    env_dict = {
        "compute-type": "BUILD_GENERAL1_SMALL",
        "image": "427443013497.dkr.ecr.us-east-2.amazonaws.com/bodo-sonar:1.0",
    }
    build_graph.append(
        {
            "identifier": "Sonar",
            "env": env_dict,
            "buildspec": buildspec,
            "depend-on": dependencies,
        }
    )


# TODO(Nick): Update this function when multiple buildspecs are included with
# nightly or another reason.
def get_buildspec_file(env_var_dict, image_path):
    return "buildscripts/aws/buildspecs/CI_buildspec.yml"


# TODO(Nick): Update this function when different build variations need
# different compute types. Most likely this will be buildspec dependent.
def get_compute_type(env_var_dict, image_path, buildspec):
    np_val = env_var_dict["NP"]
    if np_val == 1:
        return "BUILD_GENERAL1_SMALL"
    elif np_val == 2:
        return "BUILD_GENERAL1_MEDIUM"
    else:
        raise ValueError("Unsupported NP value used in buildspec")


# Function to generate the batch portion for the CI build
def generate_CI_buildspec(num_groups):
    images = {
        "linux": "427443013497.dkr.ecr.us-east-2.amazonaws.com/bodo-codebuild:3.0"
    }
    pytest_starting_marker = "not slow"
    pytest_options = [
        pytest_starting_marker + " and " + str(i) for i in range(num_groups)
    ]
    env_vars = {
        "NP": [1],
        "PYTEST_MARKER": pytest_options,
        "NUMBER_GROUPS_SPLIT": [num_groups],
    }
    return construct_batch_field(env_vars, images)


# TODO: Add support for different buildspec types (nightly, etc).
def create_new_buildspec(buildtype, num_groups):
    if buildtype == "CI":
        with open(CI_BUILDSPEC_FILENAME, "r") as f:
            existing_yaml = yaml.load(f, Loader=yaml.Loader)
        existing_yaml["batch"] = generate_CI_buildspec(num_groups)
        with open(CI_BUILDSPEC_FILENAME, "w") as f:
            # Add the necessary comments at the top.
            print(
                "# The batch section of this file is autogenerated with the script",
                file=f,
            )
            print("# in buildscripts/aws/update_buildspec_batch.py", file=f)
            print("# Any additional comments you add will not persist,", file=f)
            print("# but you can change other fields outside of batch.", file=f)
            print(
                "# Buildspec Reference: https://docs.aws.amazon.com/codebuild/latest/userguide/build-spec-ref.html",
                file=f,
            )
            print(
                "# Batch section reference: https://docs.aws.amazon.com/codebuild/latest/userguide/batch-build-buildspec.html",
                file=f,
            )
            print(
                "# Compute type options: https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-compute-types.html",
                file=f,
            )
            print(
                "# Preexisting image options: https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-available.html",
                file=f,
            )
            yaml.dump(existing_yaml, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Add more choices (e.g. Nightly)
    parser.add_argument(
        "buildtype", choices=["CI"], type=str, help="Type of build to update."
    )
    parser.add_argument(
        "num_groups", type=int, help="Number of groups to split the build into."
    )
    args = parser.parse_args()
    create_new_buildspec(args.buildtype, args.num_groups)
