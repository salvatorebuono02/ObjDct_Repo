import client
import server



if __name__ == "__main__":

    # Create the server and client objects
    server = server.Server()
    client = client.Client()

    # Load the model on the server
    profileStr = server.load_model()
    # Prepare the data on the client
    data = client.prepare_data()
    # Create the keys on the client
    context_buffer = client.create_keys(profileStr)
    # Encode the model on the server
    model_io_encoder_buf = server.encode_model(context_buffer)
    # Encrypt the data on the client
    enc_samples_buf = client.encrypt_data(model_io_encoder_buf,data)
    # Predict on the server
    enc_predictions_buf = server.predict(enc_samples_buf)
    # Decrypt the data on the client
    predictions = client.decrypt_data(enc_predictions_buf)
    # print(predictions)
