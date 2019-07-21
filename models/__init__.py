
def create_model_helper(framework, dataset, arch):
    model_helper = None
    if framework == "TensorFlow" :
        from CompFramework.models.tfnets.tfnetCreator import ModelHelper
        model_helper = ModelHelper()
    elif framework == "PyTorch":
        from CompFramework.models.ptnets.ptnetCreator import ModelHelper
        model_helper = ModelHelper()
    else:
        raise ValueError('Framework {} is not supported, use PyTorch of TensorFlow instead'.format(framework))

