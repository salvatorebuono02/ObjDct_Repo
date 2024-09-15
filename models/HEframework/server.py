import pyhelayers
from modules import utilshe
from pathlib import Path


class Server:

    batch_size = 8
    MODEL_DIR = "/home/buono/ObjDct_Repo/models/trained_models"


    def load_model(self):
        print("Loading model and preparing on server side")
        hyper_params = pyhelayers.PlainModelHyperParams()
        plain = pyhelayers.PlainModel.create(hyper_params, [self.MODEL_DIR + "/lenetfomo.onnx"])
        he_run_req = pyhelayers.HeRunRequirements()
        he_run_req.set_model_encrypted(False)
        he_run_req.set_he_context_options([pyhelayers.HeContext.create(["HEaaN_CKKS"])])
        he_run_req.optimize_for_batch_size(self.batch_size)
        profile = pyhelayers.HeModel.compile(plain, he_run_req)
        # prepare settings json to send to client so they can create the keys properly
        profileStr = profile.to_string()
        return profileStr
    
    def encode_model(self, plain, profile, context_buffer):
        print("Encoding model on server side.")
        server_context = pyhelayers.load_he_context(context_buffer)
        nn = pyhelayers.NeuralNet(server_context)
        nn.encode(plain, profile)
        print("Preparing encoder to send to client")
        model_io_encoder = pyhelayers.ModelIoEncoder(nn)
        model_io_encoder_buf = model_io_encoder.save_to_buffer()
        return model_io_encoder_buf
    
    def predict(self, server_context, nn, encrypted_samples_buf):
        print("Prediction on server side.")
        server_encrypted_samples = pyhelayers.load_encrypted_data(server_context, encrypted_samples_buf)
        enc_predictions = pyhelayers.EncryptedData(server_context)
        nn.predict(enc_predictions, server_encrypted_samples)
        enc_predictions_buf = enc_predictions.save_to_buffer()
        return enc_predictions_buf