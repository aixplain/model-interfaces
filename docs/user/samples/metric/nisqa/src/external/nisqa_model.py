import os
import datetime
import pandas as pd
import logging

pd.options.mode.chained_assignment = None
import yaml
import torch
import torch.nn as nn
from . import nisqa_lib as NL


class NisqaModel(object):
    """
    NisqaModel: Main class that loads the model and the datasets. Contains
    the training loop, prediction, and evaluation function.
    """

    def __init__(self, args):
        self.args = args

        if "mode" not in self.args:
            self.args["mode"] = "main"

        self.runinfos = {}
        self._getDevice()
        self._loadModel()
        self._loadDatasetsFile()
        self.args["now"] = datetime.datetime.today()

        if self.args["mode"] == "main":
            logging.info(yaml.dump(self.args, default_flow_style=None, sort_keys=False))

    def predict(self):
        if self.args["tr_parallel"]:
            self.model = nn.DataParallel(self.model)

        if self.args["dim"] == True:
            y_val_hat, y_val = NL.predict_dim(
                self.model,
                self.ds_val,
                self.args["tr_bs_val"],
                self.dev,
                num_workers=self.args["tr_num_workers"],
            )
        else:
            y_val_hat, y_val = NL.predict_mos(
                self.model,
                self.ds_val,
                self.args["tr_bs_val"],
                self.dev,
                num_workers=self.args["tr_num_workers"],
            )

        self.ds_val.df["model"] = self.args["name"]
        result = eval(self.ds_val.df.to_json(orient="records"))
        return result[0]["mos_pred"]

    def _loadDatasetsFile(self):
        data_dir = os.path.dirname(self.args["deg"])
        file_name = os.path.basename(self.args["deg"])
        df_val = pd.DataFrame([file_name], columns=["deg"])

        # creating Datasets ---------------------------------------------------
        self.ds_val = NL.SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir=data_dir,
            filename_column="deg",
            mos_column=None,
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            to_memory=self.args["tr_ds_to_memory"],
            to_memory_workers=self.args["tr_ds_to_memory_workers"],
            seg_hop_length=self.args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            double_ended=self.args["double_ended"],
            dim=self.args["dim"],
            filename_column_ref=None,
        )

    def _loadModel(self):
        """
        Loads the Pytorch models with given input arguments.
        """
        # if True overwrite input arguments from pretrained model
        if self.args["pretrained_model"]:
            if ":" in self.args["pretrained_model"]:
                model_path = os.path.join(self.args["pretrained_model"])
            else:
                model_path = os.path.join(os.getcwd(), self.args["pretrained_model"])
            checkpoint = torch.load(model_path, map_location=self.dev)

            if self.args["mode"] == "main":
                args_new = self.args
                self.args = checkpoint["args"]

                self.args["mode"] = args_new["mode"]
                self.args["name"] = args_new["name"]
                self.args["input_dir"] = args_new["input_dir"]
                self.args["output_dir"] = args_new["output_dir"]
                self.args["csv_file"] = args_new["csv_file"]
                self.args["csv_con"] = args_new["csv_con"]
                self.args["csv_deg"] = args_new["csv_deg"]
                self.args["csv_db_train"] = args_new["csv_db_train"]
                self.args["csv_db_val"] = args_new["csv_db_val"]
                self.args["pretrained_model"] = args_new["pretrained_model"]

                if self.args["model"] == "NISQA_DE":
                    self.args["csv_ref"] = args_new["csv_ref"]
                else:
                    self.args["csv_ref"] = None

                if self.args["model"] != "NISQA_DIM":
                    self.args["csv_mos_train"] = args_new["csv_mos_train"]
                    self.args["csv_mos_val"] = args_new["csv_mos_val"]
                else:
                    self.args["csv_mos_train"] = None
                    self.args["csv_mos_val"] = None

                self.args["tr_epochs"] = args_new["tr_epochs"]
                self.args["tr_early_stop"] = args_new["tr_early_stop"]
                self.args["tr_bs"] = args_new["tr_bs"]
                self.args["tr_bs_val"] = args_new["tr_bs_val"]
                self.args["tr_lr"] = args_new["tr_lr"]
                self.args["tr_lr_patience"] = args_new["tr_lr_patience"]
                self.args["tr_num_workers"] = args_new["tr_num_workers"]
                self.args["tr_parallel"] = args_new["tr_parallel"]
                self.args["tr_checkpoint"] = args_new["tr_checkpoint"]
                self.args["tr_bias_anchor_db"] = args_new["tr_bias_anchor_db"]
                self.args["tr_bias_mapping"] = args_new["tr_bias_mapping"]
                self.args["tr_bias_min_r"] = args_new["tr_bias_min_r"]

                self.args["tr_verbose"] = args_new["tr_verbose"]

                self.args["tr_ds_to_memory"] = args_new["tr_ds_to_memory"]
                self.args["tr_ds_to_memory_workers"] = args_new["tr_ds_to_memory_workers"]
                self.args["ms_max_segments"] = args_new["ms_max_segments"]

            elif self.args["mode"] == "predict_file":
                args_new = self.args
                self.args = checkpoint["args"]
                self.args["deg"] = args_new["deg"]
                self.args["mode"] = args_new["mode"]
                self.args["output_dir"] = args_new["output_dir"]
                self.args["pretrained_model"] = args_new["pretrained_model"]

            elif self.args["mode"] == "predict_dir":
                args_new = self.args
                self.args = checkpoint["args"]
                self.args["data_dir"] = args_new["data_dir"]
                self.args["mode"] = args_new["mode"]
                self.args["output_dir"] = args_new["output_dir"]
                self.args["pretrained_model"] = args_new["pretrained_model"]
                if args_new["bs"]:
                    self.args["tr_bs_val"] = args_new["bs"]
                if args_new["num_workers"]:
                    self.args["tr_num_workers"] = args_new["num_workers"]

            elif self.args["mode"] == "predict_csv":
                args_new = self.args
                self.args = checkpoint["args"]
                self.args["csv_file"] = args_new["csv_file"]
                self.args["mode"] = args_new["mode"]
                self.args["output_dir"] = args_new["output_dir"]
                self.args["pretrained_model"] = args_new["pretrained_model"]
                self.args["data_dir"] = args_new["data_dir"]
                self.args["input_dir"] = os.getcwd()
                self.args["csv_deg"] = args_new["csv_deg"]
                if "csv_ref" in args_new:
                    self.args["csv_ref"] = args_new["csv_ref"]
                else:
                    self.args["csv_ref"] = None
                if "csv_con" in args_new:
                    self.args["csv_con"] = args_new["csv_con"]
                if args_new["bs"]:
                    self.args["tr_bs_val"] = args_new["bs"]
                if args_new["num_workers"]:
                    self.args["tr_num_workers"] = args_new["num_workers"]

            else:
                raise NotImplementedError("Mode not available")

        if self.args["model"] == "NISQA_DIM":
            self.args["dim"] = True
        else:
            self.args["dim"] = False

        if self.args["model"] == "NISQA_DE":
            self.args["double_ended"] = True
        else:
            self.args["double_ended"] = False
            self.args["csv_ref"] = None

        # Load Model
        self.model_args = {
            "ms_seg_length": self.args["ms_seg_length"],
            "ms_n_mels": self.args["ms_n_mels"],
            "cnn_model": self.args["cnn_model"],
            "cnn_c_out_1": self.args["cnn_c_out_1"],
            "cnn_c_out_2": self.args["cnn_c_out_2"],
            "cnn_c_out_3": self.args["cnn_c_out_3"],
            "cnn_kernel_size": self.args["cnn_kernel_size"],
            "cnn_dropout": self.args["cnn_dropout"],
            "cnn_pool_1": self.args["cnn_pool_1"],
            "cnn_pool_2": self.args["cnn_pool_2"],
            "cnn_pool_3": self.args["cnn_pool_3"],
            "cnn_fc_out_h": self.args["cnn_fc_out_h"],
            "td": self.args["td"],
            "td_sa_d_model": self.args["td_sa_d_model"],
            "td_sa_nhead": self.args["td_sa_nhead"],
            "td_sa_pos_enc": self.args["td_sa_pos_enc"],
            "td_sa_num_layers": self.args["td_sa_num_layers"],
            "td_sa_h": self.args["td_sa_h"],
            "td_sa_dropout": self.args["td_sa_dropout"],
            "td_lstm_h": self.args["td_lstm_h"],
            "td_lstm_num_layers": self.args["td_lstm_num_layers"],
            "td_lstm_dropout": self.args["td_lstm_dropout"],
            "td_lstm_bidirectional": self.args["td_lstm_bidirectional"],
            "td_2": self.args["td_2"],
            "td_2_sa_d_model": self.args["td_2_sa_d_model"],
            "td_2_sa_nhead": self.args["td_2_sa_nhead"],
            "td_2_sa_pos_enc": self.args["td_2_sa_pos_enc"],
            "td_2_sa_num_layers": self.args["td_2_sa_num_layers"],
            "td_2_sa_h": self.args["td_2_sa_h"],
            "td_2_sa_dropout": self.args["td_2_sa_dropout"],
            "td_2_lstm_h": self.args["td_2_lstm_h"],
            "td_2_lstm_num_layers": self.args["td_2_lstm_num_layers"],
            "td_2_lstm_dropout": self.args["td_2_lstm_dropout"],
            "td_2_lstm_bidirectional": self.args["td_2_lstm_bidirectional"],
            "pool": self.args["pool"],
            "pool_att_h": self.args["pool_att_h"],
            "pool_att_dropout": self.args["pool_att_dropout"],
        }

        if self.args["double_ended"]:
            self.model_args.update(
                {
                    "de_align": self.args["de_align"],
                    "de_align_apply": self.args["de_align_apply"],
                    "de_fuse_dim": self.args["de_fuse_dim"],
                    "de_fuse": self.args["de_fuse"],
                }
            )

        if self.args["model"] == "NISQA":
            self.model = NL.NISQA(**self.model_args)
        elif self.args["model"] == "NISQA_DIM":
            self.model = NL.NISQA_DIM(**self.model_args)
        elif self.args["model"] == "NISQA_DE":
            self.model = NL.NISQA_DE(**self.model_args)
        else:
            raise NotImplementedError("Model not available")

        # Load weights if pretrained model is used ------------------------------------
        if self.args["pretrained_model"]:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                raise ValueError(f"Unexpected keys: {unexpected_keys}")

    def _getDevice(self):
        """
        Train on GPU if available.
        """
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")

        if "tr_device" in self.args:
            if self.args["tr_device"] == "cpu":
                self.dev = torch.device("cpu")
            elif self.args["tr_device"] == "cuda":
                self.dev = torch.device("cuda")

        if "tr_parallel" in self.args:
            if (self.dev == torch.device("cpu")) and self.args["tr_parallel"] == True:
                self.args["tr_parallel"] == False
