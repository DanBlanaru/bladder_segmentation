# bladder_segmentation
Guided Research for Bladder Segmentation from CT and PET images

- ended up not doing PET
##
Prerequisites:
- a training setup (Polyaxon / what else is currently being used)
- having write access to your NAS directory 
- having access to the raw AMOS and CT-ORG datasets in the NAS
## Explanation for each of the files:

### resize_amos and resize_ctorg 
Both do the preprocessing steps on the 3d volumes and their respective labels.
They take the pair from their initial folders, create a segmentation mask (the initial volumes for both CT-ORG and AMOS have multiple classes), and  