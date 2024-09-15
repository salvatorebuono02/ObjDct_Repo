from modelutils import *

import pyhelayers


class Client:

    FILES_DIR = '/home/buono/ObjDct_Repo/data/ShipDataset'
    transform = Compose([transforms.Resize((88, 88)), transforms.ToTensor()])


    def prepare_data(self):
        test_dataset = ShipDataset(
            root_dir=self.FILES_DIR,
            transform=self.transform,
            train=False
        )

        test_img_list=[]
        for image,label,boxes in test_dataset:
            test_img_list.append(image)
            if len(test_img_list)==8:
                break
        return test_img_list
    

    def create_keys(self, profileStr):
        """
        Receives the profileStr from the server and creates the keys. 
        Save the context. Note that this saves all the HE library information, including the 
        public key, allowing the server to perform HE computations.
        The secret key is not saved here, so the server won't be able to decrypt.
        The secret key is never stored unless explicitly requested by the user using the designated 
        method.
        """
        print("Creating keys on client side")
        client_profile = pyhelayers.HeProfile()
        client_profile.from_string(profileStr)
        client_context = pyhelayers.HeModel.create_context(client_profile)
        context_buffer = client_context.save_to_buffer()
        return context_buffer

    def encrypt_data(self, client_context, model_io_encoder_buf, test_img_list):
        """
        Encrypts the data. 
        Load io encoder
        """
        print("Encrypting data on client side.")
        client_model_io_encoder = pyhelayers.load_io_encoder(client_context, model_io_encoder_buf)
        encrypted_samples = pyhelayers.EncryptedData(client_context)
        client_model_io_encoder.encode_encrypt(encrypted_samples, [test_img_list])
        encrypted_samples_buf = encrypted_samples.save_to_buffer()
        return encrypted_samples_buf
    
    def decrypt_data(self, client_context, enc_predictions_buf, client_model_io_encoder):
        """
        Decrypts the data.
        Prameters:
            client_context: the context
            enc_predictions_buf: the encrypted predictions
            client_model_io_encoder: the model io encoder
        Returns:
            plain_predictions: the decrypted predictions
        """

        print("Decrypt on client side")
        client_enc_predictions=pyhelayers.load_encrypted_data(client_context,enc_predictions_buf)
        plain_predictions = client_model_io_encoder.decrypt_decode_output(client_enc_predictions)
        print('predictions',plain_predictions)



