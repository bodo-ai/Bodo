#!/bin/bash

### Default values
BODO_VERSION_FILE="./bodo-platform-image/bodo_versions.txt"

help() {
   # Display Help
   echo "Packer build images script"
   echo "--help           Show this help output."
   echo
   echo "Options required:"
   echo "--image-sha            Image SHA"
   echo "--node-role            Node role, default: worker."
   echo
   echo "Options optional:"
   echo "--bodo-version-file    Path to file with bodo versions, default: ../bodo_versions.txt."
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
    --bodo-version-file)
      BODO_VERSION_FILE="$2"
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
echo "--bodo-version-file: $BODO_VERSION_FILE"


# Check if all required values are set
if [[ -z $IMAGE_SHA || -z $NODE_ROLE || -z $BODO_VERSION_FILE  ]]; then
  echo 'Error: One or more variables are undefined'
  help
  exit 1
fi

echo "Starting script to create an image definition for specialized linux images..."
ind=0
while read -r LINE; do
    echo "Iteration $ind"
    
    bodo_version=$(echo $LINE | awk '{print $1}')
    bodo_version_short=${bodo_version:0:60}
    bodo_version_short="${bodo_version_short/+/-}" # replace illegal + with legal '-'

    img_name="bodo_${NODE_ROLE}_${IMAGE_SHA}_${bodo_version_short}"
    echo "bodo_version: $bodo_version"
    echo "bodo_version_short: $bodo_version_short"
    echo "img_name: $img_name"

    echo "Start create image definition"
        az sig image-definition create -g bodo-images-resource-grp \
        --gallery-name BodoImages \
        --gallery-image-definition "${img_name}"  \
        --publisher Bodo \
        --offer "${NODE_ROLE}" \
        --sku "${IMAGE_SHA}_${bodo_version_short}" \
        --os-type linux \
        --hyper-v-generation V2

    echo "Finished create image definition..."
    (( ind++ ))
    echo "Moving to next bodo version..."
done < "${BODO_VERSION_FILE}"

echo "All done..."
