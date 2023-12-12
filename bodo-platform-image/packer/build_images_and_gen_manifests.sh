#!/bin/bash

### Default values
NODE_ROLE="worker"
TEMPLATE_FILE="templates/images.pkr.hcl"
BODO_VERSION_FILE="../bodo_versions.txt"
SKIP_CREATE_IMAGE=false

help() {
   # Display Help
   echo "Packer build images script"
   echo "--help           Show this help output."
   echo
   echo "Options required:"
   echo "--image-sha            Image SHA"
   echo "--node-role            Node role, default: worker."
   echo "--playbook-file        Path to Packer playbook file."
   echo
   echo "Options optional:"
   echo "--bodo-version-file    Path to file with bodo versions, default: ../bodo_versions.txt."
   echo "--skip-create-images   If this option is set we will skip create images for AWS/Azure."
}


while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      help
      exit 0
      ;;
    --image-sha)
      IMAGE_SHA="$2"
      shift
      shift
      ;;
    --node-role)
      NODE_ROLE="$2"
      shift
      shift
      ;;
    --playbook-file)
      PLAYBOOK_FILE="$2"
      shift
      shift
      ;;
    --skip-create-images)
      SKIP_CREATE_IMAGE=true
      shift
      ;;
    --bodo-version-file)
      BODO_VERSION_FILE="$2"
      shift
      shift
      ;;
    --build-name)
      BUILD_NAME="$2"
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo "Script parameters ..."
echo "--image-sha: $IMAGE_SHA"
echo "--node-role: $NODE_ROLE"
echo "--playbook-file: $PLAYBOOK_FILE"
echo "--bodo-version-file: $BODO_VERSION_FILE"
echo "--skip-create-images: $SKIP_CREATE_IMAGE"
echo "--build-name: $BUILD_NAME"

# Check if all required values are set
if [[ -z $IMAGE_SHA || -z $NODE_ROLE || -z $PLAYBOOK_FILE || -z $BODO_VERSION_FILE || -z $SKIP_CREATE_IMAGE ]]; then
  echo 'Error: One or more variables are undefined'
  help
  exit 1
fi

echo "Starting script build images using packer..."
ind=0
exitCode=0
while read -r LINE; do
    echo "Iteration: $ind"

    bodo_version=$(echo $LINE | awk '{print $1}')
    bodo_version_short=${bodo_version:0:60}
    bodo_version_short="${bodo_version_short/+/-}" # replace illegal + with legal -
    echo "bodo_version: $bodo_version"
    echo "bodo_version_short: $bodo_version_short"

    echo "Packer Init / Download Plugins ..."
    packer init -upgrade "${TEMPLATE_FILE}"

    echo "Packer build image..."
    packer build -force \
      -var "node_role=${NODE_ROLE}" \
      -var "playbook_file=${PLAYBOOK_FILE}" \
      -var "image_sha=${IMAGE_SHA}" \
      -var "bodo_version=${bodo_version}" \
      -var "bodo_version_short=${bodo_version_short}" \
      -var "build_name=${BUILD_NAME}" \
      -var "skip_create_image=${SKIP_CREATE_IMAGE}" \
      -var "ind=${ind}" \
      "${TEMPLATE_FILE}"

    newExitCode=$?
    echo "Finished creating VM Image..."

    if [ -f "${NODE_ROLE}_manifest_${ind}.json" ]; then
      echo "Manifest File:"
      cat "${NODE_ROLE}_manifest_${ind}.json"
    fi
    
    # Exit code is 0 if all inputs succeed
    # See https://stackoverflow.com/questions/16357624/anding-exit-codes-in-bash
    ! (( newExitCode || exitCode ))
    exitCode=$?
    (( ind++ ))
    echo "Moving to next bodo version..."
done < "${BODO_VERSION_FILE}"

echo "All done..."
exit $exitCode
