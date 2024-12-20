"""
   Copyright 2023 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
import random
import tensorflow as tf
import keras.backend as K
import os
from cosine_annealing import CosineAnnealingScheduler
import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
wandb.login(key="3c7b273814544590b64c54d9a5242bde38616e02")
wandb.init(project='gnn', name='test_pkts_enhancedk4_repeat')

# Run eagerly-> Turn true for debugging only
RUN_EAGERLY = False

tf.config.run_functions_eagerly(RUN_EAGERLY)


def _reset_seeds(seed: int = 42) -> None:
    """Reset rng seeds, and also indicate tf if to run eagerly or not

    Parameters
    ----------
    seed : int, optional
        Seed for rngs, by default 42
    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
      super(LRLogger, self).__init__()
      self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
      lr = self.optimizer.get_config()["learning_rate"]
      wandb.log({"lr": lr}, commit=True)
def moe_loss(y_true, y_pred):
    # diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    # loss  = 100.0 * K.mean(diff, axis=-1)
    # print("loss_val", 10e-2* y_pred[-1])
    return  100 * tf.math.reduce_mean(tf.abs((y_true - y_pred[:-1]) / y_true))   + 4 * y_pred[-1] * 0.0


def moe_metric(y_true, y_pred):
    # diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    # loss  = 100.0 * K.mean(diff, axis=-1)
    # print("loss_val", 10e-2* y_pred[-1])
    return  100 * tf.math.reduce_mean(tf.abs((y_true - y_pred[:-1]) / y_true))
 

def get_default_callbacks() -> List[tf.keras.callbacks.Callback]: 
    """Returns the default callbacks for the training of the models
    (EarlyStopping and ReduceLROnPlateau callbacks)
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.0001,
            start_from_epoch=4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            verbose=1,
            mode="min",
            min_delta=0.001,
        ),

        # CosineAnnealingScheduler(T_max=10, eta_max=0.001, eta_min=1e-6),

        # tf.keras.callbacks.LearningRateScheduler(tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate))
    ]


def get_default_hyperparams() -> Dict[str, Any]:
    """Returns the default hyperparameters for the training of the models. That is
    - Adam optimizer with lr=0.001
    - MeanAbsolutePercentageError loss
    - No additional metrics
    - EarlyStopping and ReduceLROnPlateau callbacks
    - 100 epochs
    """
    # schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    # initial_learning_rate=0.001,
    # first_decay_steps=3380*10,
    # t_mul=1.0,
    # m_mul=0.5,
    # alpha=0.0
    # )
    return {
        "optimizer": tf.keras.optimizers.AdamW(learning_rate=0.001),#5.0000e-04 #0.001
        "loss": tf.keras.losses.MeanAbsolutePercentageError(),
        # "loss": moe_loss,

        "metrics": [],
        "additional_callbacks": get_default_callbacks(),
        "epochs": 150,
    }


def get_min_max_dict(  
    ds: tf.data.Dataset, params: List[str], include_y: Optional[str] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get the min and the max-min for different parameters of a dataset. Later used by the models for the min-max normalization.

    Parameters
    ----------
    ds : tf.data.Dataset
        Training dataset where to base the min-max normalization from.

    params : List[str]
        List of strings indicating the parameters to extract the features from.

    include_y : Optional[str], optional
        Indicates if to also extract the features of the output variable.
        Inputs indicate the string key used on the return dict. If None, it is not included.
        By default None.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing the values needed for the min-max normalization.
        The first value is the min value of the parameter, and the second is 1 / (max - min).
    """

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    sample, label = next(iter_ds)
    params_lists = {param: sample[param].numpy() for param in params}
    if include_y:
        params_lists[include_y] = label.numpy()

    # Include the rest of the samples
    for sample, label in iter_ds:
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    for param, param_list in params_lists.items():
        min_val = np.min(param_list, axis=0)
        min_max_val = np.max(param_list, axis=0) - min_val
        if min_max_val.size == 1 and min_max_val == 0:
            scores[param] = [min_val, 0]
            print(
                f"Min-max normalization Warning: {param} has a max-min of 0.")
        elif min_max_val.size > 1 and np.any(min_max_val == 0):
            min_max_val[min_max_val != 0] = 1 / min_max_val[min_max_val != 0]
            scores[param] = [min_val, min_max_val]
            print(
                f"Min-max normalization Warning: Several values of {param} has a max-min of 0."
            )
        else:
            scores[param] = [min_val, 1 / min_max_val]

    return scores


def train_and_evaluate(
    ds_path: Union[str, Tuple[str, str]],
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics: List[tf.keras.metrics.Metric],
    additional_callbacks: List[tf.keras.callbacks.Callback],
    epochs: int = 100,
    ckpt_path: Optional[str] = None,
    tensorboard_path: Optional[str] = None,
    restore_ckpt: bool = False,
    final_eval: bool = True,
) -> Tuple[tf.keras.Model, Union[float, np.ndarray, None], Union[dict, None]]:
    """
    Train the given model with the given dataset, using the provided parameters
    Besides for defining the hyperparameters, refer to get_default_hyperparams()

    Parameters
    ----------
    ds_path : str
        Path to the dataset. Datasets are expected to be in tf.data.Dataset format, and to be compressed with GZIP.
        If ds_path is a string, then it used as the path to both the training and validation dataset.
        If so, it is expected that the training and validation datasets are located in "{ds_path}/training" and "{ds_path}/validation" respectively.
        If ds_path is a tuple of two strings, then the first string is used as the path to the training dataset,
        and the second string is used as the path to the validation dataset.

    model : tf.keras.Model
        Instance of the model to train. Besides being a tf.keras.Model, it should have the same constructor and the name parameter
        as the models in the models module.

    optimizer : tf.keras.Optimizer
        Optimizer used by the training process

    loss : tf.keras.losses.Loss
        Loss function to be used by the process

    metrics : List[tf.keras.metrics.Metric]
        List of additional metrics to consider during training

    additional_callbacks : List[tf.keras.callbacks.Callback], optional
        List containing tensorflow callback functions to be added to the training process.
        A callback to generate tensorboard and checkpoint files at each epoch is already added.

    epochs : int, optional
        Number of epochs of in the training process, by default 100

    ckpt_path : Optional[str], optional
        Path where to store the training checkpints, by default "{repository root}/ckpt/{model name}"

    tensorboard_path : Optional[str], optional
        Path where to store tensorboard logs, by default "{repository root}/tensorboard/{model name}"

    restore_ckpt : bool, optional
        If True, before training the model, it is checked if there is a checkpoint file in the ckpt_path.
        If so, the model loads the latest checkpoint and continues training from there. By default False.

    final_eval : bool, optional
        If True, the model is evaluated on the validation dataset one last time after training, by default True

    Returns
    -------
    Tuple[tf.keras.Model, Union[float, np.ndarray, None]]
        Instance of the trained model, and the result of its evaluation
    """

    # Reset tf state
    _reset_seeds()
    # Check epoch number is valid
    assert epochs > 0, "Epochs must be greater than 0"
    # Load ds
    if isinstance(ds_path, str):
        ds_train = tf.data.Dataset.load(
            f"{ds_path}/training", compression="GZIP")
        ds_val = tf.data.Dataset.load(
            f"{ds_path}/validation", compression="GZIP")
    else:
        ds_train = tf.data.Dataset.load(ds_path[0], compression="GZIP")
        ds_val = tf.data.Dataset.load(ds_path[1], compression="GZIP")
    # Checkpoint path
    if ckpt_path is None:
        ckpt_path = f"ckpt/{model.name}"
    # Tensorboard path
    if tensorboard_path is None:
        tensorboard_path = f"tensorboard/{model.name}"

    # Apply min-max normalization
    # model.set_min_max_scores(get_min_max_dict(
    #     ds_train, model.min_max_scores_fields))
    
    model.set_mean_std_scores(get_mean_std_dict(ds_train, model.mean_std_scores_fields))
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        run_eagerly=RUN_EAGERLY,
    )
    # Load checkpoint
    # restore_ckpt = True
    # ckpt_path = "/home/ec2-user/ckpt/Baseline_mb_attn"
    if restore_ckpt:
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        if ckpt is not None:
            print("Restoring from checkpoint")
            model.load_weights(ckpt)
        else:
            print(
                f"WARNING: No checkpoint was found at '{ckpt_path}', training from scratch instead..."
            )
    else:
        print("restore_ckpt = False, training from scratch")

    # Create callbacks
    cpkt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_path, "{epoch:02d}-{val_loss:.4f}"),
        verbose=1,
        mode="min",
        save_best_only=False,
        save_weights_only=True,
        save_freq="epoch",
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_path, histogram_freq=1
    )

    # Train model
    History = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        # callbacks=[cpkt_callback, tensorboard_callback] + additional_callbacks,
        callbacks=[cpkt_callback, tensorboard_callback, wandb.keras.WandbCallback(save_model=False),  LRLogger(optimizer)] + additional_callbacks,

        use_multiprocessing=True,
    )

    if final_eval:
        return model, model.evaluate(ds_val), History.history
    else:
        return model, None, None
    
def get_mean_std_dict(
    ds: tf.data.Dataset, params: List[str], include_y: Optional[str] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get the min and the max-min for different parameters of a dataset. Later used by the models for the min-max normalization.

    Parameters
    ----------
    ds : tf.data.Dataset
        Training dataset where to base the min-max normalization from.

    params : List[str]
        List of strings indicating the parameters to extract the features from.

    include_y : Optional[str], optional
        Indicates if to also extract the features of the output variable.
        Inputs indicate the string key used on the return dict. If None, it is not included.
        By default None.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing the values needed for the min-max normalization.
        The first value is the min value of the parameter, and the second is 1 / (max - min).
    """

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    sample, label = next(iter_ds)
    params_lists = {param: sample[param].numpy() for param in params}
    if include_y:
        params_lists[include_y] = label.numpy()

    # Include the rest of the samples
    for sample, label in iter_ds:
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    for param, param_list in params_lists.items():
        mean_val = np.mean(param_list, axis=0)
        std_val = np.std(param_list, axis=0)
        
        if all(std_val) == 0:
            scores[param] = [mean_val, std_val]
        else:
            scores[param] = [mean_val, 1/std_val]

    return scores
if __name__ == "__main__":
    import argparse
    # import the training model
    import models_advanced_pkts as models
    import std_models_with_moe_mlpgate_old_mb

    parser = argparse.ArgumentParser(
        description="Train a model for flow delay prediction"
    )
    parser.add_argument(
        "-ds", type=str, help="Either 'CBR+MB' or 'MB'", required=True)
    parser.add_argument(
        "-cfv", action="store_true", help="Perform cross-fold validation"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for cross-fold validation. Default is 5. Ignored if -cf is not set",
    )
    args = parser.parse_args()

    # Check the scenario
    if args.ds == "CBR+MB":
        # the dataset with feature ipg and onRate
        ds_path = "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/final_combined_cv"
        model = models.Baseline_cbr_mb_std

        print("args.ds", args.ds)
    elif args.ds == "MB":
        # the dataset with feature ipg、ibg、onRate and packetperBurst
        ds_path = "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/cbr_test_only"
        # model = models.Baseline_mb_attn
        model = std_models_with_moe_mlpgate_old_mb.Baseline_cbr_MoE
    else:
        raise ValueError("Unrecognized dataset")

    # code for simple training/validation
    if not args.cfv:
        _reset_seeds()
        trained_model, evaluation = train_and_evaluate(
            # os.path.join(ds_path, "3"), model(), 
            ("/home/ec2-user/gnnet-ch23-dataset-cbr-mb/final_combined", "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/cbr_test_only_comb" ),
            # ("/home/ec2-user/gnnet-ch23-dataset-cbr-mb/mb_extracted", "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/mb_test_only" ),
            # ("/home/ec2-user/gnnet-ch23-dataset-cbr-mb/mb_extracted_full", "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/mb_test_only_comb2" ),


            # ("/home/ec2-user/gnnet-ch23-dataset-cbr-mb/cbr_test_delay_indx_1", "/home/ec2-user/gnnet-ch23-dataset-cbr-mb/cbr_test_delay_indx_1" ),


            model(), 

            **get_default_hyperparams(),
        )
        print("Final evaluation:", evaluation)

    # code for cross-fold validation
    else:
        trained_models = []
        trained_models_val_loss = []
        ckpt_path = f"ckpt/{model.name}_ipg_onRate_cv/"
        tensorboard_path = f"tensorboard/{model.name}_ipg_onRate_cv/"

        # Execute each fold
        for fold_idx in range(args.n_folds):
            print("***** Fold", fold_idx, "*****")
            _reset_seeds()
            trained_model, evaluation, history = train_and_evaluate(
                os.path.join(ds_path, str(fold_idx)),
                model(),
                **get_default_hyperparams(),
                ckpt_path=os.path.join(ckpt_path, str(fold_idx)),
                tensorboard_path=os.path.join(tensorboard_path, str(fold_idx)),

            )
            trained_models.append(trained_model)
            trained_models_val_loss.append(evaluation)

        # Print final evaluation
        for fold_idx, evaluation in enumerate(trained_models_val_loss):
            print(f"Fold {fold_idx} evaluation:",
                  trained_models_val_loss[fold_idx])