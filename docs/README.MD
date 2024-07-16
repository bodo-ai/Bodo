## Clone the bodo platform repository and set required env vars

> [!WARNING]
> You need to have https://github.com/Bodo-inc/bodo-platform cloned and updated. Ensure you are on the `main` branch.
> You need to set the PLATFORM_SDK_PATH env var to the path of the `sdk2` folder in the bodo-platform repo.

- `export PLATFORM_SDK_PATH=<path_to_sdk2_folder_in_bodo_platform_repo/>`

## Create virtualenv and install requirements
- `make pip-install`

> [!NOTE]
> Any package you do not find can be pip installed.


## Building docs locally 
- to render docs, use `make mkdocs-build`. 


## Serving docs locally 
- to render docs, use `make mkdocs-serve`. 
- docs will be rendered on : http://127.0.0.1:8000/ 
- your terminal needs to be running while docs are being served. 

## Creating and api docs 

**WARNING** : Only do this if you are modifying API reference. Contact @ritwika314 first.

**note** This process is a bit hacky and manual at the moment, but will improve with time. 

- To install our small custom markdown api extension: `python setup.py develop`
- To generate the formatted API files : `make gen_api`
 
## Editing diagrams: 

Diagrams are stored in : https://drive.google.com/drive/u/0/folders/1lt7AHGKMPVmJyuEjaoEqme5kFEDgAYJ5 
You can open them using https://app.diagrams.net/ . 
Generate an svg, with a transparent background and embed fonts for any diagrams. 


## Deploying:

`mike` is used for deployment. 
- To deploy the docs version you are building locally, use `make mike-deploy version_name=<your_version_name>`.
- To view the deployed docs locally, use `make mike-serve`.
- To push the current update docs to the docs site, use `make mike-push version_name=<your_version_name>`.
- To publish new release docs to the docs site, use `make mike-push-latest version_name=<your_version_name>`.

