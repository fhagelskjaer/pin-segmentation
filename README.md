# pin-segmentation
### Code for generating training data for the paper:
### In-Hand Pose Estimation and Pin Inspection for Insertion of Through-Hole Components


First download all through-hole components from:

https://www.pcb-3d.com/membership_type/free/

Then convert them to PLY files and place them in the folder "data/".


The script will use the file "segmentation_index.json" to generate ".h5" files with test and training information.

>		python generate_dataset_from_json.py
