#!/usr/bin/env python

import pyhelayers
from pathlib import Path
import client
import numpy as np
print('Imported pyhelayers version ',pyhelayers.VERSION)

# You can change these variables to point to your own model
# and data files.
# Also, you can see how this model was created and trained in folder data_gen 


batch_size=8

# 1. Server side: load model and prepare it for work under encryption
print("Loading model and preparing on server side")
hyper_params = pyhelayers.PlainModelHyperParams()
plain = pyhelayers.PlainModel.create(hyper_params, [str(Path(__file__).parent.parent) + "/trained_models/lenetfomo.onnx"])
he_run_req = pyhelayers.HeRunRequirements()
he_run_req.set_model_encrypted(False)
he_run_req.set_he_context_options([pyhelayers.HeContext.create(["HEaaN_CKKS"])])
he_run_req.optimize_for_batch_size(batch_size)
profile = pyhelayers.HeModel.compile(plain, he_run_req)
# prepare settings json to send to client so they can create the keys properly
profileStr=profile.to_string()

# 2. Client side
# Receives profileStr
print("Creating keys on client side")
client_profile=pyhelayers.HeProfile()
client_profile.from_string(profileStr)
# Creates context with keys
client_context = pyhelayers.HeModel.create_context(client_profile)
# Save the context. Note that this saves all the HE library information, including the 
# public key, allowing the server to perform HE computations.
# The secret key is not saved here, so the server won't be able to decrypt.
# The secret key is never stored unless explicitly requested by the user using the designated 
# method.
context_buffer = client_context.save_to_buffer()


# 3. Server side
print("Encoding model on server side.")
server_context=pyhelayers.load_he_context(context_buffer)
nn = pyhelayers.NeuralNet(server_context)
nn.encode(plain, profile)
print("Preparing encoder to send to client")
model_io_encoder = pyhelayers.ModelIoEncoder(nn)
model_io_encoder_buf= model_io_encoder.save_to_buffer()

# 4. Client side
print("Encrypting data on client side.")
client=client.Client()
plain_samples = client.prepare_data()
plain_samples = np.array(plain_samples)
print('Loaded samples of shape',plain_samples.shape)
# Load io encoder
client_model_io_encoder=pyhelayers.load_io_encoder(client_context,model_io_encoder_buf)
encrypted_samples = pyhelayers.EncryptedData(client_context)
client_model_io_encoder.encode_encrypt(encrypted_samples, [plain_samples])
encrypted_samples_buf=encrypted_samples.save_to_buffer()

# 5. server side
print("Prediction on server side.")
server_encrypted_samples=pyhelayers.load_encrypted_data(server_context,encrypted_samples_buf)
enc_predictions = pyhelayers.EncryptedData(server_context)
nn.predict(enc_predictions, server_encrypted_samples)
enc_predictions_buf=enc_predictions.save_to_buffer()

# 6. client side
print("Decrypt on client side")
client_enc_predictions=pyhelayers.load_encrypted_data(client_context,enc_predictions_buf)
plain_predictions = client_model_io_encoder.decrypt_decode_output(client_enc_predictions)
print('predictions',plain_predictions)