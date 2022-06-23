for i in "$@"
  do
    case $i in
      --taskname=*)
      TASKNAME="${i#*=}"
      shift
      ;;
    esac
  done

#mkdir ./build/
#cd ./build/

#cp -r ../model_registry.${TASKNAME}.json ./model_registry.json

RUN_ID="0ff6a17a4cb948d99b801ae41b7339dd" #$(cat ./model_registry.json | jq -r '.[-1].model.run_id')
ARTIFACT_PATH="model.onnx" #$(cat ./model_registry.json | jq -r '.[-1].model.artifact_path')
DATABRICKS_HOST="https://outreach-databricks-managed.cloud.databricks.com/" #$(cat ./model_registry.json | jq -r '.[-1].model.databricks_host')
MLFLOW_VERSION="1.23.1" #$(cat ./model_registry.json | jq -r '.[-1].model.mlflow_version')
NUM_LABELS="27" #$(cat ./model_registry.json | jq -r '.[-1].model.num_labels')
MODEL_TYPE="roberta" #$(cat ./model_registry.json | jq -r '.[-1].model.model_type')

echo "run id                = $RUN_ID" #RUN_ID
echo "artifact path         = $ARTIFACT_PATH" #ARTIFACT_PATH
echo "databricks host       = $DATABRICKS_HOST" #DATABRICKS_HOST
echo "mlflow version        = $MLFLOW_VERSION" #MLFLOW_VERSION
echo "num labels            = $NUM_LABELS" #NUM_LABELS
echo "model type            = $MODEL_TYPE" #MODEL_TYPE

sudo pip install -U mlflow==$MLFLOW_VERSION

export MLFLOW_TRACKING_URI=databricks
export DATABRICKS_HOST=$DATABRICKS_HOST

DOWNLOAD_PATH=$(mlflow artifacts download --run-id $RUN_ID --artifact-path $ARTIFACT_PATH)
mkdir -p ./tmp/model/
mv $DOWNLOAD_PATH ./tmp/model/

cp -r ../model_repository/$MODEL_TYPE ./model_repository
rm -r ./model_repository/get-preds-pytorch/
cp ./tmp/model/$ARTIFACT_PATH/model.onnx ./model_repository/get-preds-onnx/1/
cp ../src/tpm/segmentation.py ./model_repository/get-sents/1/model.py
cp ../src/tpm/tokenization.py ./model_repository/get-tokens/1/model.py
cp ../src/tpm/postprocess.py ./model_repository/get-postproc/1/model.py
cp ./tmp/model/$ARTIFACT_PATH/labels.txt ./model_repository/get-postproc/1/labels.txt

cp ./model_repository/get-preds-onnx/config.pbtxt.template ./model_repository/get-preds-onnx/config.pbtxt
cp ./model_repository/get-postproc/config.pbtxt.template ./model_repository/get-postproc/config.pbtxt
SEARCH="@NUM_LABELS"
REPLACE=$NUM_LABELS
sed -i "s/$SEARCH/$REPLACE/" ./model_repository/get-preds-onnx/config.pbtxt
sed -i "s/$SEARCH/$REPLACE/" ./model_repository/get-postproc/config.pbtxt
cat ./model_repository/get-preds-onnx/config.pbtxt
cat ./model_repository/get-postproc/config.pbtxt

cp -r ../Dockerfile ./
docker build -t stevezheng23/tritonserver:21.10-py3-classifier .

docker run --rm -d --shm-size 1g \
-p8000:8000 -p8001:8001 -p8002:8002 \
stevezheng23/tritonserver:21.10-py3-classifier \
tritonserver --model-repository=/model_repository

sleep 10
curl -v localhost:8000/v2/health/ready
