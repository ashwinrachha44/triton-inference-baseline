import os
import torch
import triton_python_backend_utils as pd_utils

from transformers import AutoModelForSequenceClassification

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

        model_dir = os.path.join(args['model_repository'], args['model_version'], 'model')
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.is_cuda = torch.cuda.is_available()
        self.model = model.cuda() if self.is_cuda else model
        self.model.eval()
    
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
            input_ids = pd_utils.get_input_tensor_by_name(request, 'input_ids')
            input_ids = torch.from_numpy(input_ids.as_numpy())
            attention_mask = pd_utils.get_input_tensor_by_name(request, 'attention_mask')
            attention_mask = torch.from_numpy(attention_mask.as_numpy())
            encoded_input = {
              'input_ids': input_ids.cuda() if self.is_cuda else input_ids,
              'attention_mask': attention_mask.cuda() if self.is_cuda else attention_mask
            }
            token_type_ids = pd_utils.get_input_tensor_by_name(request, 'token_type_ids')
            if token_type_ids is not None:
              token_type_ids = torch.from_numpy(token_type_ids.as_numpy())
              encoded_input["token_type_ids"] = token_type_ids.cuda() if self.is_cuda else token_type_ids
            with torch.no_grad():
              output = self.model(**encoded_input)
            output_0 = output[0].detach().cpu().numpy() if self.is_cuda else output[0].detach().numpy()
            output_0 = pd_utils.Tensor('output_0', output_0)
            response = pd_utils.InferenceResponse(output_tensors=[output_0])
            responses.append(response)
        
        return responses
