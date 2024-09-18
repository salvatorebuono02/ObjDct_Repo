import pyhelayers
import numpy as np
from pathlib import Path


class Server:

    batch_size = 8
    MODEL_DIR = "/home/buono/ObjDct_Repo/models/trained_models"


    def __init__(self):
        self.profile = None
        self.plain = None
        self.nn = None
        self.model_io_encoder = None
        self.server_context = None


    def load_model(self):
        print("Loading model and preparing on server side")
        hyper_params = pyhelayers.PlainModelHyperParams()
        self.plain = pyhelayers.PlainModel.create(hyper_params, [str(Path(__file__).parent.parent) + "/trained_models/lenetfomo.onnx"])
        he_run_req = pyhelayers.HeRunRequirements()
        he_run_req.set_model_encrypted(False)
        he_run_req.set_he_context_options([pyhelayers.HeContext.create(["HEaaN_CKKS"])])
        he_run_req.optimize_for_batch_size(self.batch_size)
        self.profile = pyhelayers.HeModel.compile(self.plain, he_run_req)
        # prepare settings json to send to client so they can create the keys properly
        profileStr = self.profile.to_string()
        return profileStr
    
    def encode_model(self, context_buffer):
        print("Encoding model on server side.")
        self.server_context = pyhelayers.load_he_context(context_buffer)
        self.nn = pyhelayers.NeuralNet(self.server_context)
        self.nn.encode(self.plain, self.profile)
        print("Preparing encoder to send to client")
        self.model_io_encoder = pyhelayers.ModelIoEncoder(self.nn)
        model_io_encoder_buf = self.model_io_encoder.save_to_buffer()
        return model_io_encoder_buf
    
    def predict(self, encrypted_samples_buf):
        print("Prediction on server side.")
        server_encrypted_samples = pyhelayers.load_encrypted_data(self.server_context, encrypted_samples_buf)
        enc_predictions = pyhelayers.EncryptedData(self.server_context)
        self.nn.predict(enc_predictions, server_encrypted_samples)
        enc_predictions_buf = enc_predictions.save_to_buffer()
        return enc_predictions_buf