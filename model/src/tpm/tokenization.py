import os
import triton_python_backend_utils as pd_utils

from transformers import AutoTokenizer, TensorType

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

        tokenizer_dir = os.path.join(args['model_repository'], args['model_version'], 'tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
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
            texts = pd_utils.get_input_tensor_by_name(request, 'text').as_numpy().tolist()
            texts = [t.decode('utf-8') for t in texts]
            tokens = self.tokenizer(texts,
                                    padding="longest",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors=TensorType.NUMPY)
            output_tensors = [
              pd_utils.Tensor('input_ids', tokens['input_ids']),
              pd_utils.Tensor('attention_mask', tokens['attention_mask'])
            ]
            if "token_type_ids" in tokens:
              output_tensors.append(pd_utils.Tensor('token_type_ids', tokens['token_type_ids']))
            response = pd_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)
        
        return responses
