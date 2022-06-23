import os
import triton_python_backend_utils as pd_utils

import numpy as np

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

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

        nltk.download('punkt')
        self.sent_tokenizer = PunktSentenceTokenizer()
    
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
            sent_texts, sent_offsets, sent_lengths = [], [], []
            sent_indices, group_indices = [], []
            for i, text in enumerate(texts):
              sents = self.sent_tokenizer.span_tokenize(text)
              for j, (s, e) in enumerate(sents):
                sent = text[s:e]
                offset = s
                length = len(sent)
                sent_texts.append(sent)
                sent_offsets.append(offset)
                sent_lengths.append(length)
                sent_indices.append(j)
                group_indices.append(i)

            output_tensors = [
              pd_utils.Tensor('sent_text', np.array(sent_texts, dtype=object)),
              pd_utils.Tensor('sent_offset', np.array(sent_offsets, dtype=int)),
              pd_utils.Tensor('sent_length', np.array(sent_lengths, dtype=int)),
              pd_utils.Tensor('sent_index', np.array(sent_indices, dtype=int)),
              pd_utils.Tensor('group_index', np.array(group_indices, dtype=int))
            ]
            response = pd_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)
        
        return responses
