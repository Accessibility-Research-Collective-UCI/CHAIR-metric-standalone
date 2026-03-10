import json
import os
import pickle
from chair import CHAIR
from tqdm.auto import tqdm

import nltk

nltk.download("punkt_tab")

# unfortunately I think hat data must be a global variable
# load data
hat_file = "../partial-abstainer/data/mscoco/mscoco-500-baselines_03-03-26.json"
with open(hat_file, "r", encoding="utf-8") as f:
    hat_data = json.load(f)

# ========================================================================================================================
# FUNCTIONS
# ========================================================================================================================


def image_id_from_filename(filename):
    """Gets the image id from the MS COCO filename.

    Args:
        filename (str): The MS COCO filename. Ex: COCO_val2014_000000016903.jpg

    Returns:
        str: The image id. Ex: 16903
    """
    return str(int(filename.split(".")[0].split("_")[-1]))


def compute_chair_overall(chair_scored_list):
    """
    Compute CHAIR metrics for a list of CHAIR scores.
    """
    # variables to hold
    num_caps = 0.0
    num_hallucinated_caps = 0.0
    hallucinated_word_count = 0.0
    coco_word_count = 0.0

    num_recall_gt_objects = 0.0
    num_gt_objects = 0.0

    for chair_output in chair_scored_list:
        num_caps += 1
        coco_word_count += len(chair_output["mscoco_generated_words"])
        num_hallucinated_caps += (
            1 if len(chair_output["mscoco_hallucinated_words"]) > 0 else 0
        )
        hallucinated_word_count += len(chair_output["mscoco_hallucinated_words"])
        num_gt_objects += len(chair_output["mscoco_gt_words"])
        num_recall_gt_objects += len(chair_output["recall_gt_objects"])

    # compute overall metrics
    chair_s = num_hallucinated_caps / num_caps
    chair_i = hallucinated_word_count / coco_word_count
    # add
    recall = num_recall_gt_objects / num_gt_objects if num_gt_objects > 0 else 0

    return {
        "CHAIRs": chair_s,
        "CHAIRi": chair_i,
        "Recall": recall,
    }


# Step 1 and 2: Compute original or baseline CHAIR
def get_vlm_chair(vlm: str, chair_type: str = "CHAIRs"):
    """
    Computes the given CHAIR metric for the given VLM.

    Parameters
    ----------
    vlm : float
        The VLM on which to simulate thresholding.
    chair_type : str
        Default "CHAIRs". The specific CHAIR metric to look for (one of "CHAIRi", "CHAIRs", or "Recall").

    Returns
    -------
    chair_score : float
        The CHAIR score requested.
    """
    chair_vlm_dataset = dict()

    # collect atomics and image ids
    all_atomics = []
    all_image_ids = []
    empty_image_ids = []
    for dat in hat_data:
        # atomics
        atomics = [
            atom["statement"] for atom in dat["captions"][vlm]["atomics"]["reference"]
        ]
        all_atomics += atomics

        # image ids
        image_id = int(image_id_from_filename(dat["file_name"]))
        all_image_ids += [image_id for _ in atomics]

        # catch
        if len(atomics) == 0:
            empty_image_ids.append(image_id)

    # compute chair
    chair_out = chair_evaluator.compute_chair(
        all_atomics, all_image_ids, compact_output=True
    )

    return chair_out["overall_metrics"][chair_type]


# Step 3: threshold simulation


# helper function: create dataset to compute CHAIR given the VLM
def _collect_vlm_chair_scores(vlm: str):
    """
    Helper function for simulate_thresholding(). Creates dataset with CHAIR scores for each atomic.

    Parameters
    ----------
    vlm : str
        The VLM for which we will compute CHAIR

    Returns
    -------
    chair_vlm_dataset : dict
        dataset with chair scores
    """
    # I'm just gonna brute force it
    chair_vlm_dataset = dict()

    # collect atomics and image ids
    all_atomics = []
    all_image_ids = []
    empty_image_ids = []
    for dat in hat_data:
        # atomics
        atomics = [
            atom["statement"] for atom in dat["captions"][vlm]["atomics"]["reference"]
        ]
        all_atomics += atomics

        # image ids
        image_id = int(image_id_from_filename(dat["file_name"]))
        all_image_ids += [image_id for _ in atomics]

        # catch
        if len(atomics) == 0:
            empty_image_ids.append(image_id)

    # compute chair
    chair_out = chair_evaluator.compute_chair(
        all_atomics, all_image_ids, compact_output=True
    )

    # parse output
    sentence_output = chair_out["sentences"]
    for out in sentence_output:
        # add new key if necessary
        image_key = out["image_id"]
        if image_key not in chair_vlm_dataset.keys():
            # print(f"{image_key} not found. Adding...")
            chair_vlm_dataset[image_key] = {}
            for chair_type in ["CHAIRi", "CHAIRs", "Recall"]:
                chair_vlm_dataset[image_key][chair_type] = []

        # append scores to list in the dict
        for chair_type in ["CHAIRi", "CHAIRs", "Recall"]:
            chair_vlm_dataset[image_key][chair_type] += [out["metrics"][chair_type]]

    # deal with empty atomic lists
    for image_key in empty_image_ids:
        chair_vlm_dataset[image_key] = {}
        for chair_type in ["CHAIRi", "CHAIRs", "Recall"]:
            chair_vlm_dataset[image_key][chair_type] = []

    # now it's all saved, so we should be ok to return
    # we could do additional parsing here or maybe later
    return chair_vlm_dataset


def _collect_vlm_atomic_similarity(vlm: str):
    """
    Helper function for simulate_thresholding(). Returns atomic similarity for the given VLM.
    """
    # ummmmm dict of lists?
    atomic_sim_scores = {}
    for dat in hat_data:
        image_id = int(image_id_from_filename(dat["file_name"]))

        # early quit
        if "atomic_similarity" not in dat["captions"][vlm].keys():
            continue
        curr_sim_scores = [
            atom_sim["avg_similarity_score"]
            for atom_sim in dat["captions"][vlm]["atomic_similarity"]
        ]
        atomic_sim_scores[image_id] = curr_sim_scores

    return atomic_sim_scores


def _get_simulated_scores(threshold, vlm, atomic_sim_scores, chair_type):
    """
    Helper function for simulate_thresholding(). Returns average CHAIR score based on parsed data and the average % atomics remaining.
    """
    # ok planning
    # go through each image in atomic_sim_scores via keys
    # threshold and remember indices where 1?
    # call CHAIR again idk

    # prep CHAIR input
    kept_atomics = []
    total_atomics = 0
    all_image_ids = []

    for dat in hat_data:
        img_id = int(image_id_from_filename(dat["file_name"]))
        curr_atom_sims = atomic_sim_scores[img_id]

        atomics = [
            atom["statement"] for atom in dat["captions"][vlm]["atomics"]["reference"]
        ]
        if len(curr_atom_sims) != len(atomics):
            # huh?
            # print(f"Atomic similarity count mismatch! {len(curr_atom_sims)} != {len(atomics)} - {img_id}")
            continue

        curr_kept = [
            atom for i, atom in enumerate(atomics) if curr_atom_sims[i] >= threshold
        ]

        # update CHAIR input
        kept_atomics += curr_kept
        all_image_ids += [img_id for _ in curr_kept]

        # update counter
        total_atomics += len(atomics)

    # compute chair
    chair_out = chair_evaluator.compute_chair(
        kept_atomics, all_image_ids, compact_output=True
    )

    return chair_out["overall_metrics"][chair_type], len(kept_atomics) / total_atomics


def _threshold_searching(
    thresholds, target_chair, atomic_sim_scores, vlm, chair_type, prev_n_thresholds=None
):
    """
    Helper function for simulate_thresholding(). Performs binary search on the list of thresholds so we don't search linearly. Returns the final threshold, avg chair, and pct atomics remaining.

    WE ARE ASSUMING CHAIR DECREASES MONOTONICALLY WITH THRESHOLD <- hm maybe a bad assumption lol i hope this doesnt break
    """
    n_thresholds = len(thresholds)

    # print(f"{n_thresholds} thresholds...")

    if n_thresholds == 1:
        # base case - we found it
        avg_chair, pct_remaining = _get_simulated_scores(
            thresholds[0], vlm, atomic_sim_scores, chair_type
        )
        return thresholds[0], avg_chair, pct_remaining
    if prev_n_thresholds == n_thresholds:
        # drop the biggest threshold? idk it should be close
        return _threshold_searching(
            thresholds[:-1],
            target_chair,
            atomic_sim_scores,
            vlm,
            chair_type,
            n_thresholds,
        )

    median_idx = int(n_thresholds / 2)
    test_threshold = thresholds[median_idx]

    # test
    avg_chair, _ = _get_simulated_scores(
        test_threshold, vlm, atomic_sim_scores, chair_type
    )

    # and check
    if avg_chair <= target_chair:
        # then a higher threshold will lead to a lower CHAIR
        # so we cut off everything higher than test_threshold
        return _threshold_searching(
            thresholds[: (median_idx + 1)],
            target_chair,
            atomic_sim_scores,
            vlm,
            chair_type,
            n_thresholds,
        )
    else:
        # then a lower threshold will lead to an even high CHAIR
        # this is not desired, so we cut off everything lower
        return _threshold_searching(
            thresholds[(median_idx + 1) :],
            target_chair,
            atomic_sim_scores,
            vlm,
            chair_type,
            n_thresholds,
        )


def simulate_thresholding(chair_score: float, vlm: str, chair_type: str = "CHAIRi"):
    """
    Simulates thresholding on the input vlm CHAIR scores to meet the threshold and get the % atomics remaining

    Parameters
    ----------
    chair_score : float
        The score for which to aim.
    vlm : str
        The VLM on which to simulate thresholding.
    chair_type : str
        Default `"CHAIRs"`. The specific CHAIR metric to look for (one of `"CHAIRi"`, `"CHAIRs"`, or `"Recall"`).

    Returns
    -------
    threshold : float
        The threshold at which the simulation on the VLM meets the given threshold for the given CHAIR score.
    pct_atomics : float
        The percent of remaining atomics
    """
    # value checking
    if chair_type.lower() not in ["chairi", "chairs", "recall"]:
        raise ValueError(
            f"""Parameter chair_type expected one of ["CHAIRi", "CHAIRs", "Recall"] but got {chair_type}."""
        )
    # correct the chair metric string
    corrections = {"chairi": "CHAIRi", "chairs": "CHAIRs", "recall": "Recall"}
    chair_type = corrections[chair_type.lower()]

    # look at the data from the saved dictionary
    chair_data = _collect_vlm_chair_scores(vlm)

    # parse & grab relevant CHAIR score
    parsed_chair_data = {}
    for dat in hat_data:
        image_id = int(image_id_from_filename(dat["file_name"]))
        parsed_chair_data[image_id] = chair_data[image_id][chair_type]

    # grab atomic similarity scores
    atomic_sim_scores = _collect_vlm_atomic_similarity(vlm)

    # ok uh try to simulate?
    # maybeeeeee sort the thresholds and go from there
    all_thresholds = []
    for sim_list in atomic_sim_scores.values():
        all_thresholds += sim_list
    all_thresholds = sorted(all_thresholds)

    # move up the list
    thres, avg_chair, pct_remaining = _threshold_searching(
        all_thresholds, chair_score, atomic_sim_scores, vlm, chair_type
    )
    print(
        f"{vlm} {chair_type}: {avg_chair:.04f} <= {chair_score:.04f} at threshold {thres:.04f} (min threshold={all_thresholds[0]:.04f})"
    )
    # we want the matching threshold and pct remaining
    return thres, pct_remaining


# ========================================================================================================================
# EXPERIMENTS
# ========================================================================================================================
def save_scores_and_simulation(chair_score_types, base_vlms, baselines):
    chair_scores = {}
    simulations = {}
    # execute
    for chair_type in tqdm(chair_score_types, desc="CHAIR Score Type", leave=True):
        chair_scores[chair_type] = {}
        simulations[chair_type] = {}

        out_str += f"{chair_type}\n"
        for vlm in tqdm(base_vlms, desc="Base VLM", leave=False):
            # base chair for comparison
            base_chair = get_vlm_chair(vlm, chair_type)
            chair_scores[chair_type][vlm] = base_chair

        # it should have gone through all 4 base VLMS
        for baseline_vlm in tqdm(list(baselines.keys()), desc="Baselines", leave=False):
            # base chair for comparison
            base_chair = get_vlm_chair(baseline_vlm, chair_type)
            chair_scores[chair_type][baseline_vlm] = base_chair

            base_vlm = baselines[baseline_vlm]

            # and run the simulation threshold
            matched_thres, pct_remaining = simulate_thresholding(
                base_chair, base_vlm, chair_type
            )
            simulations[chair_type][baseline_vlm] = [matched_thres, pct_remaining]

    # save the pickles
    with open("chair_scores.pkl", "wb") as f:
        pickle.dump(chair_scores, f)

    with open("simulations.pkl", "wb") as f:
        pickle.dump(simulations, f)


def save_plotting_data(base_vlms, chair_score_types):
    # save the data
    plotting_data = {}

    for vlm in tqdm(base_vlms, desc="VLM"):
        plotting_data[vlm] = {}

        # look at the data from the saved dictionary
        chair_data = _collect_vlm_chair_scores(vlm)

        for chair_type in tqdm(chair_score_types, desc="CHAIR Type"):
            plotting_data[vlm][chair_type] = {}
            # parse & grab relevant CHAIR score
            parsed_chair_data = {}
            for dat in hat_data:
                image_id = int(image_id_from_filename(dat["file_name"]))
                parsed_chair_data[image_id] = chair_data[image_id][chair_type]

            # grab atomic similarity scores
            atomic_sim_scores = _collect_vlm_atomic_similarity(vlm)

            # ok uh try to simulate?
            # maybeeeeee sort the thresholds and go from there
            all_thresholds = []
            for sim_list in atomic_sim_scores.values():
                all_thresholds += sim_list
            all_thresholds = sorted(all_thresholds)

            all_avg_chairs = []
            all_pcts_remaining = []
            # move up the list
            for thres in tqdm(all_thresholds, desc="Thresholding...", leave=False):
                avg_chair, pct_remaining = _get_simulated_scores(
                    thres, vlm, atomic_sim_scores, chair_type
                )

                all_avg_chairs.append(avg_chair)
                all_pcts_remaining.append(pct_remaining)

            # save the data
            plotting_data[vlm][chair_type]["thresholds"] = all_thresholds
            plotting_data[vlm][chair_type]["pcts_remaining"] = all_pcts_remaining
            plotting_data[vlm][chair_type]["chair_scores"] = all_avg_chairs

    # save in pickle
    with open("plotting_data.pkl", "wb") as f:
        pickle.dump(plotting_data, f)


# ========================================================================================================================
# MAIN
# ========================================================================================================================

if __name__ == "__main__":
    # global chair
    coco_path = "coco_annotations"
    cached_instance = "chair.pkl"
    chair_evaluator = None
    if os.path.exists(cached_instance):
        chair_evaluator = pickle.load(open(cached_instance, "rb"))
        print(f"loaded evaluator from cache: {cached_instance}", flush=True)
    else:
        print("cache not setted or not exist yet, building from scratch...", flush=True)
        chair_evaluator = CHAIR(coco_path)
        pickle.dump(chair_evaluator, open(cached_instance, "wb"))
        print(f"cached evaluator to: {cached_instance}", flush=True)

    # ========================================================================================================================
    # CONSTANTS
    # ========================================================================================================================

    vlms = [
        # 'gpt', 'gemini', 'llama', 'molmo',
        "llava_hf",
        "llava_orig",
        "minigpt_llama",
        "minigpt_vicuna",
        "lure",
        "marine",
        "reverse_t0.003",
        "reverse_t0.0003",
        "fact_rlhf",
        "nullu_llava",
        "nullu_minigpt",
        "halc_llava",
        "halc_minigpt4",
    ]

    base_vlms = [
        "llava_orig",
        "llava_hf",
        "minigpt_llama",
        "minigpt_vicuna",
    ]

    baselines = {
        "lure": "minigpt_vicuna",
        "marine": "llava_hf",
        "reverse_t0.003": "llava_orig",
        "reverse_t0.0003": "llava_orig",
        "fact_rlhf": "llava_orig",
        "nullu_llava": "llava_orig",
        "nullu_minigpt": "minigpt_llama",
        "halc_llava": "llava_orig",
        "halc_minigpt4": "minigpt_llama",
    }

    chair_score_types = ["CHAIRi", "CHAIRs"]

    # ok yay hopefully everything runs :D
    save_scores_and_simulation(chair_score_types, base_vlms, baselines)
    save_plotting_data(base_vlms, chair_score_types)
