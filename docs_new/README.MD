## Requirements

- `pip install mkdocs`
- `pip install mkdocs-awesome-pages-plugin`
- `pip install mkdocs-autorefs`
- `pip install mkdocs-material`
- `pip install mkdocs-material-extensions`
- `pip install mike`

**note** Any package you do not find can be pip installed.

## Building docs locally 

- to render docs, use `mkdocs serve`. 
- docs will be rendered on : http://127.0.0.1:8000/ 
- your terminal needs to be running while docs are being served. 

## Creating and api docs 

**WARNING** : Only do this if you are modifying API reference. Contact @ritwika314 first.

**note** This process is a bit hacky and manual at the moment, but will improve with time. 

- To install our small custom markdown api extension: `python setup.py develop`
- To generate the formatted API files : `python apigen.py`
 
## Editing diagrams: 

Diagrams are stored in : https://drive.google.com/drive/u/0/folders/1lt7AHGKMPVmJyuEjaoEqme5kFEDgAYJ5 
You can open them using https://app.diagrams.net/ . 
Generate an svg, with a transparent background and embed fonts for any diagrams. 


## Deploying:

`mike` is used for deployment. 
