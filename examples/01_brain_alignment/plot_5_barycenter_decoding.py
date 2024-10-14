# %%
# Fetch haxby dataset
from nilearn import datasets, image
import pandas as pd

subject_list = [1, 2, 3]
haxby_dataset = datasets.fetch_haxby(subjects=subject_list)
mask_img = haxby_dataset.mask


def load_haxby_data(subject):
    fmri_filename = haxby_dataset.func[subject - 1]
    labels = pd.read_csv(haxby_dataset.session_target[subject - 1], sep=" ")
    # Keep only the data corresponding to faces or house
    conditions = ["face", "house"]
    labels = labels[labels["labels"].isin(conditions)]
    # Get the indices of the corresponding images
    mask_idx = labels.index.values
    # Extract the corresponding volume
    volume = image.load_img(fmri_filename)
    selcted_img = image.index_img(volume, mask_idx)
    return selcted_img, labels


subject_data = {}
for subject in subject_list:
    img, labels = load_haxby_data(subject)
    subject_data[subject] = (img, labels)
