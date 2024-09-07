import pyhelayers


if __name__=='__main__':

    # nnp = pyhelayers.NeuralNetPlain()
    # hyper_params = pyhelayers.PlainModelHyperParams()
    # nnp.init_from_files(hyper_params, ["/home/buono/ObjDct_Repo/models/trained_models/lenetfomo.onnx"])

    # # PYHELAYERS
    # he_run_req = pyhelayers.HeRunRequirements()
    # he_run_req.set_he_context_options([pyhelayers.DefaultContext()])
    # he_run_req.optimize_for_batch_size(8)

    # profile = pyhelayers.HeModel.compile(nnp, he_run_req)
    # batch_size = profile.get_optimal_batch_size()
    # print('Profile ready. Batch size=',batch_size)

    # context = pyhelayers.HeModel.create_context(profile)
    # print('HE context initalized')

    # # print(context.get_he_config_requirement())
    # # print(context.has_secret_key())
    # pub_functions = context.get_public_functions()
    # print('Public functions:',pub_functions)


    he_run_req = pyhelayers.HeRunRequirements()
    he_run_req.set_he_context_options([pyhelayers.DefaultContext()])
    he_run_req.optimize_for_batch_size(8)

    nn = pyhelayers.NeuralNet()
    nn.encode_encrypt(["/home/buono/ObjDct_Repo/models/trained_models/lenetfomo.onnx"], he_run_req)

    
    context = nn.get_created_he_context()

    pub_functions = context.get_public_functions()
    print('Public functions:',pub_functions)



