# %%
# Fetch haxby dataset
from nilearn import datasets, image, maskers
import pandas as pd
import numpy as np
from fugw.mappings import FUGWSparseBarycenter
from fugw.datasets import fetch_vol_geometry
from fugw.scripts import coarse_to_fine

# %%
subject_list = [1, 2, 3]
resolution = 10
haxby_dataset = datasets.fetch_haxby(subjects=subject_list)
mask_img = datasets.load_mni152_gm_mask(resolution=resolution)
masker = maskers.NiftiMasker(mask_img=mask_img, standardize=True).fit()


# %%
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
    selected_img = image.index_img(volume, mask_idx)
    return selected_img, labels


subject_data = []
for subject in range(len(subject_list)):
    img, labels = load_haxby_data(subject)
    subject_dict = {"img": img, "labels": labels}
    subject_data.append(subject_dict)

# %%
features_list = [
    masker.transform(subject_data[i]["img"]) for i in range(len(subject_data))
]
n_voxels = features_list[0].shape[1]
weights_list = [
    np.ones(n_voxels) / n_voxels for _ in range(len(features_list))
]
geometry_embedding, d_max = fetch_vol_geometry(
    "mni152_gm_mask", resolution, rank=3, method="euclidean"
)
n_samples = 100
mesh_sample = coarse_to_fine.sample_volume_uniformly(
    segmentation=image.get_data(mask_img),
    embeddings=geometry_embedding,
    n_samples=n_samples,
)
geometry_embedding_normalized = geometry_embedding / d_max

# %%
fugw_barycenter = FUGWSparseBarycenter(
    alpha_coarse=0.5,
    alpha_fine=0.5,
    rho_coarse=float("inf"),
    rho_fine=float("inf"),
    eps_coarse=1.0,
    eps_fine=1.0,
)
fugw_barycenter.fit(
    weights_list,
    features_list,
    geometry_embedding_normalized,
    mesh_sample=mesh_sample,
    coarse_mapping_solver_params={"nits_bcd": 3, "nits_uot": 100},
    fine_mapping_solver_params={"nits_bcd": 3, "nits_uot": 100},
    nits_barycenter=1,
    device="cpu",
    verbose=True,
)
