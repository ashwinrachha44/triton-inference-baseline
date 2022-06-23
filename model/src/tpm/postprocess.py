import os
import numpy as np
import triton_python_backend_utils as pd_utils
from scipy.special import softmax

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        label_file = os.path.join(args['model_repository'], args['model_version'], 'labels.txt')
        with open(label_file, 'r', encoding='utf-8') as in_file:
            self.label_list = in_file.read().splitlines()
    
    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            predict_scores = pd_utils.get_input_tensor_by_name(request, 'input_0').as_numpy()
            predict_scores = softmax(predict_scores, axis=-1)
            predict_indices = np.argmax(predict_scores, axis=-1).astype(np.int32).tolist()
            predict = np.array([self.label_list[idx] for idx in predict_indices], dtype=object)
            score = np.array([predict_scores[i][idx] for i, idx in enumerate(predict_indices)], dtype=np.float32)
            predict = pd_utils.Tensor('predict', predict)
            score = pd_utils.Tensor('score', score)
            response = pd_utils.InferenceResponse(output_tensors=[predict, score])
            responses.append(response)
        
        return responses
