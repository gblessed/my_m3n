import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable, List
import tensorflow as tf
import os
import sys
#python3 /home/ec2-user/ITU-ML5G-PS-007-GNN-m0b1us/predict_separate.py --te-path "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/mb_test_comb"
#python3 /home/ec2-user/ITU-ML5G-PS-007-GNN-m0b1us/predict_separate.py --te-path "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/mb_extracted_test"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _default_individual_prediction(model: tf.keras.Model, sample: any) -> np.ndarray:
    """
    Default function to predict the flow delay values for a given sample. The prediction
    is returned as a flat, unnormalized numpy array and in seconds.

    Parameters
    ----------
    model : tf.keras.Model
        The model to use for prediction. Its weights should be already trained.
    sample_features : any
        The sample to predict. By default it expects samples from a tf.data.Dataset,
        but can be any format that allows to iterate over it.


    Returns
    -------
    np.ndarray
        The predictions to return.
    """
    # Obtain the prediction as numpy array, and flatten
    pred = model(sample).numpy().reshape((-1,))
    # Transform the prediction from ms to s, and return
    return pred / 1000

def sample_type(sample_features):
    """_summary_
    Funciton to determine the type of sample for separate predictions
    Args:
        sample_features (_type_): any
        The sample to determine the type

    Returns:
        str: return the type of sample, 'CBR+MB' or 'MB'
    """
    flow_types = list(
        map(lambda x: x, sample_features["flow_type"].numpy()))
    all_MB = True
    for flow_type in flow_types:
        if flow_type[0] == 1:
            all_MB = False
    if all_MB:
        return 'MB'
    else:
        return 'CBR+MB'


def predict(
    ds: any,
    cbr_mb_model: tf.keras.Model,
    mb_model: tf.keras.Model,
    predict_file_name: str,
    submission_verification_file_path: Optional[
        str
    ] = "/home/ec2-user/ITU-ML5G-PS-007-GNN-m0b1us/submission_verification.txt",
    predict_file_path: Optional[str] = "/home/ec2-user/ITU-ML5G-PS-007-GNN-m0b1us",
    individual_prediction: Callable[[any, any],
                                    np.ndarray] = _default_individual_prediction,
    verbose: bool = False,
) -> None:
    """Use the given model to predict flow delay values for the given dataset.
    Predictions are stored in a csv file, which itself will be compressed with zip.

    Parameters
    ----------
    ds : any
        Object representing the loaded test dataset. By default it expects the
        tf.data.Dataset format, but can be any format that allows to iterate over it.

    cbr_mb_model : tf.keras.Model
        (CBR+MB)tf.keras.Model instance to use for prediction. Its weights should be already
        trained.

    mb_model : tf.keras.Model
        (MB)tf.keras.Model instance to use for prediction. Its weights should be already
        trained.

    predict_file_name : str,
        Name of the generated file with all the predictions

    submission_verification_file_path : Optional[str], optional
        Path to the file which contains the data to verify the submission.
        This file is provided by the challenge. There is one version of the file for
        the toy dataset, and another one for test dataset. By default, it points to the
        test dataset verification file.

    predict_file_path : Optional[str], optional
        Path to the directory where to store the predictions. By default, the prediction
        are stored in the current directory.

    verbose : bool, optional
        If True, print additional information, by default False
    """
    list_sample_file_id = []
    list_flow_id = []
    list_predicted_delay = []
    num_sample_file_id = []
    num_flow_id = []
    num_predicted_delay = []
    # list_flow_type = []
    if verbose:
        print()
    for ii, (sample_features, _) in enumerate(iter(ds)):
        if verbose:
            print(f"\r Progress: {ii} / {len(ds)}", end="")
        # For each sample, fill in the fields
        sample_file_id = sample_features["sample_file_id"].numpy().tolist()
        list_sample_file_id += sample_file_id
        num_sample_file_id.append(len(sample_file_id))

        flow_id = list(
            map(lambda x: x.decode(), sample_features["flow_id"].numpy()))
        list_flow_id += flow_id
        num_flow_id.append(len(flow_id))
        # determine the type of sample for predicting separately
        if (sample_type(sample_features) == 'CBR+MB'):
            model = cbr_mb_model
        else:
            model = mb_model

        predicted_delay = individual_prediction(
            model, sample_features).tolist()
        list_predicted_delay += predicted_delay
        num_predicted_delay.append(len(predicted_delay))
    if verbose:
        print()

    # Verify the submission
    with open("/home/ec2-user/ITU-ML5G-PS-007-GNN-m0b1us/submission_verification.txt", "r") as f:
        flows_per_sample = list(map(lambda x: int(x.strip()), f.readlines()))
        total_flows = sum(flows_per_sample)
    success = True

    # 1. Check total number of flows is correct
    if len(list_sample_file_id) != total_flows:
        print_err(
            f"ERROR: When counting the number of sample files id, the number of flows "
            + f"is incorrect. Expected {total_flows}, got {len(list_sample_file_id)}"
        )
        success = False
    if len(list_flow_id) != total_flows:
        print_err(
            f"ERROR: When counting the number of flows id, the number of flows is "
            + f"incorrect. Expected {total_flows}, got {len(list_flow_id)}"
        )
        success = False
    if len(list_predicted_delay) != total_flows:
        print_err(
            f"ERROR: When counting the number of predicted delays, the number of flows "
            + f"is incorrect. Expected {total_flows}, got {len(list_predicted_delay)}"
        )
        success = False

    # 2. Check the number of flows per sample is correct
    for ii, (num_sample, num_flow, num_delay, true_flows) in enumerate(
        zip(num_sample_file_id, num_flow_id,
            num_predicted_delay, flows_per_sample)
    ):
        num_sample_ver = num_sample != true_flows
        num_flow_ver = num_flow != true_flows
        num_delay_ver = num_delay != true_flows

        if num_sample_ver or num_flow_ver or num_delay_ver:
            err_msg = f"ERROR: The number of flows for sample {ii} is incorrect."
            if num_sample_ver:
                err_msg += f" Expected {true_flows} sample file ids, got {num_sample}."
            if num_flow_ver:
                err_msg += f" Expected {true_flows} flow ids, got {num_flow}."
            if num_delay_ver:
                err_msg += f" Expected {true_flows} predicted delays, got {num_delay}."
            print_err(err_msg)
            success = False

    if success is False:
        print_err(
            f"WARNING: The submission is not correct. Please check the errors above. "
            + f"Exiting..."
        )
        sys.exit(1)
    # Save predictions using pandas
    print("Verification passed! Saving predictions...")

    df = pd.DataFrame(
        {
            "sample_file_id": list_sample_file_id,
            "flow_id": list_flow_id,
            "predicted_delay": list_predicted_delay,
        }
    )
    if predict_file_path is not None:
        os.makedirs(predict_file_path, exist_ok=True)
        zip_path = f"{os.path.join(predict_file_path, predict_file_name)}.zip"
    else:
        zip_path = f"{predict_file_name}.zip"

    df.to_csv(
        zip_path,
        index=False,
        sep=",",
        header=False,
        float_format="%.9f",
        compression={"method": "zip",
                     "archive_name": f"{predict_file_name}.csv"},
    )
    return None


if __name__ == "__main__":
    import argparse
    # import models_predict as models
    import models_advanced as models

    from train import get_mean_std_dict, get_min_max_dict

    parser = argparse.ArgumentParser(
        description="Use a trained model to generate predictions from"
    )
    parser.add_argument(
        "--te-path",
        type=str,
        help="Path to test dataset (to generate the predictions from)",
        required=True,
    )
    args = parser.parse_args()

    # new change
    # load CBR+MB model
    cbr_mb_model = models.Baseline_cbr_mb_std()
    cbr_mb_ckpt_path = "/home/ec2-user/ckpt/Baseline_cbr_mb_std"
    cbr_mb_tr_path =  "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/final_combined"
    cbr_mb_model.set_mean_std_scores(
        get_mean_std_dict(
            tf.data.Dataset.load(cbr_mb_tr_path, compression="GZIP"),
            # cbr_mb_model.min_max_scores_fields,
            cbr_mb_model.mean_std_scores_fields,

        )
    )
    cbr_mb_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
    )

    cbr_mb_ckpt = tf.train.latest_checkpoint(cbr_mb_ckpt_path)
    cbr_mb_model.load_weights(cbr_mb_ckpt)
    # load MB model
    # mb_model = models.Baseline_mb()
    # mb_ckpt_path =    "/home/ec2-user/ckpt/Baseline_mb_attn"
    # mb_tr_path = "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/mb_extracted_cv/0/training"
    # mb_model.set_min_max_scores(
    #     get_min_max_dict(
    #         tf.data.Dataset.load(mb_tr_path, compression="GZIP"),
    #         mb_model.min_max_scores_fields,
    #     )
    # )


    mb_model = models.Baseline_mb_attn()
    mb_ckpt_path =    "/home/ec2-user/ckpt/Baseline_mb_attn"
    mb_tr_path = "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/mb_extracted"
    mb_model.set_mean_std_scores(
        get_mean_std_dict(
            tf.data.Dataset.load(mb_tr_path, compression="GZIP"),
            # mb_model.min_max_scores_fields,
            mb_model.mean_std_scores_fields,

        )
    )
    mb_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
    )
    mb_ckpt = tf.train.latest_checkpoint(mb_ckpt_path)
    mb_model.load_weights(mb_ckpt)
    # new change

    # Select correct verification file
    ver_file_path = ("verification_files/submission_verification.txt")

    # Load the test dataset
    ds = tf.data.Dataset.load(args.te_path, compression="GZIP")
    # Predict muti-type sample by muti-model
    predict(ds, cbr_mb_model, mb_model, "predictions", ver_file_path)