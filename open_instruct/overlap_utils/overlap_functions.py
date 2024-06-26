from collections import defaultdict
import torch
from dadapy._cython import cython_overlap as c_ov
from .extract_repr import extract_activations
from .pairwise_distances import compute_distances
import sys
import numpy as np
import warnings


# def _get_nn_indices(
#     self,
#     coordinates,
#     distances,
#     dist_indices,
#     k,
# ):

#     if k > self.maxk:
#         if dist_indices is None and distances is not None:
#             # if we are given only a distance matrix without indices we expect it to be in square form
#             assert distances.shape[0] == distances.shape[1]
#             _, dist_indices, _, _ = self._init_distances(distances, k)
#             return dist_indices, k
#         elif coordinates is not None:
#             # if coordinates are available and k > maxk distances should be recomputed
#             # and nearest neighbors idenitified up to k.
#             _, dist_indices = compute_nn_distances(
#                 coordinates, k, self.metric, self.period
#             )
#             return dist_indices, k
#         else:
#             # we must set k=self.maxk and continue the compuation
#             warnings.warn(
#                 f"Chosen k = {k} is greater than max available number of\
#                 nearest neighbors = {self.maxk}. Setting k = {self.maxk}",
#                 stacklevel=2,
#             )
#             k = self.maxk

#     if dist_indices is not None:
#         # if nearest neighbors are available (up to maxk) return them
#         return dist_indices, k

#     elif distances is not None:
#         # otherwise if distance matrix in square form is available find the first k nearest neighbors
#         _, dist_indices, _, _ = self._init_distances(distances, k)
#         return dist_indices, k
#     else:
#         # otherwise compute distances and nearest neighbors up to k.
#         _, dist_indices = compute_nn_distances(coordinates, k, self.metric, self.period)
#         return dist_indices, k


def return_data_overlap(
    indices_base,
    indices_other,
    subjects,
    k=30,
):
    # """Return the neighbour overlap between the full space and another dataset.

    # An overlap of 1 means that all neighbours of a point are the same in the two spaces.

    # Args:
    #     coordinates (np.ndarray(float)): the data set to compare, of shape (N , dimension of embedding space)
    #     distances (np.ndarray(float), tuple(np.ndarray(float), np.ndarray(float)) ):
    #                                 Distance matrix (see base class for shape explanation)
    #     k (int): the number of neighbours considered for the overlap

    # Returns:
    #     (float): the neighbour overlap of the points
    # """
    # assert any(
    #     var is not None for var in [self.X, self.distances, self.dist_indices]
    # ), "MetricComparisons should be initialized with a dataset."

    # assert any(
    #     var is not None for var in [coordinates, distances, dist_indices]
    # ), "The overlap with data requires a second dataset. \
    #     Provide at least one of coordinates, distances, dist_indices."

    # dist_indices_base, k_base = self._get_nn_indices(
    #     self.X, self.distances, self.dist_indices, k
    # )

    # dist_indices_other, k_other = self._get_nn_indices(
    #     coordinates, distances, dist_indices, k
    # )

    assert indices_base.shape[0] == indices_other.shape[0]
    # k = min(k_base, k_other)
    ndata = indices_base.shape[0]

    overlaps_full = c_ov._compute_data_overlap(
        ndata, k, indices_base.astype(int), indices_other.astype(int)
    )

    # overlaps = {}
    # for subject in np.unique(subjects):
    #     mask = subject == subjects
    #     overlaps[subject] = np.mean(overlaps_full[mask])

    return np.mean(overlaps_full)


@torch.no_grad()
def compute_overlap(
    accelerator,
    model,
    val_loader,
    tokenizer,
    target_layers,
    embdims,
    dtypes,
    base_indices,
    subjects,
    results_dir,
    filename,
    ckpt_dir,
):
    target_layer_names = list(target_layers.values())
    name_to_idx = {val: key for key, val in target_layers.items()}

    model.eval()
    extr_act = extract_activations(
        accelerator=accelerator,
        model=model,
        dataloader=val_loader,
        target_layers=target_layer_names,
        embdim=embdims,
        dtypes=dtypes,
        use_last_token=True,
        print_every=200,
    )

    extr_act.extract(val_loader, tokenizer)
    extr_act.remove_hooks()

    accelerator.print("representations extracted")
    sys.stdout.flush()

    act_dict = extr_act.hidden_states

    overlaps = defaultdict(dict)
    for i, (name, act) in enumerate(act_dict.items()):
        torch.save(act, f"{results_dir}/{name}{filename}.pt")

    print(f"{results_dir}/{name}{filename}.pt\n")

    # dirpath_actual = (
    #     "/u/area/ddoimo/ddoimo/finetuning_llm/open-instruct/results/llama-2-7b"
    # )
    # actual = torch.load(
    #     f"{dirpath_actual}/base_model.model.model.layers.0.input_layernorm_outepoch2.pt"
    # )

    # print(dirpath_actual)

    # dirpath_original = "/u/area/ddoimo/ddoimo/open/geometric_lens/repo/results/validation/llama-2-7b/0shot"
    # print("{dirpath original}")
    # expected = torch.load(f"{dirpath_original}/l0_hook_output_target.pt")

    # torch.testing.assert_close(actual, expected)

    # print("actual and original match\n")

    for shots in base_indices.keys():
        # for norm in base_indices[shots].keys():
        # ov_tmp = defaultdict(dict)
        ov_tmp = defaultdict()
        for i, (name, act) in enumerate(act_dict.items()):
            act = act.to(torch.float64).numpy()

            if name_to_idx[name] < 1:
                continue
            else:
                _, dist_index, _, _ = compute_distances(
                    X=act,
                    n_neighbors=40 + 1,
                    n_jobs=1,
                    working_memory=2048,
                    range_scaling=40 + 1,
                    argsort=False,
                )

                ov_tmp[name] = return_data_overlap(
                    indices_base=dist_index,
                    indices_other=base_indices[shots][name_to_idx[name]],
                    subjects=subjects,
                    k=30,
                )


        overlaps[shots] = ov_tmp

    model.train()
    return overlaps
