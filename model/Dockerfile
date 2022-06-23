FROM nvcr.io/nvidia/tritonserver:22.05-py3
LABEL author.name="Ashwin Rachha"
LABEL author.email="ashwin.rachha@outreach.io"

RUN pip --no-cache-dir install torch==1.8.1
RUN pip --no-cache-dir install sentencepiece==0.1.94
RUN pip --no-cache-dir install transformers==4.12.0
RUN pip --no-cache-dir install scipy==1.8.0
RUN pip --no-cache-dir install nltk==3.6.7

COPY model_repository /model_repository
