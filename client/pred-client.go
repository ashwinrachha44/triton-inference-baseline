package client

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"time"

	"triton-inference-baseline/dto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	triton "triton-inference-baseline/client/nvidia_inferenceserver"
)

type TritonPredClient interface {
	GetPreds(texts []string) []dto.ClsResult
	IsReady() bool
}

type tritonPredClient struct {
	grpcClient   triton.GRPCInferenceServiceClient
	modelName    string
	modelVersion string
}

func ClientPrint() {
	fmt.Println("Everything is working well")
}

func CreatePredClient(serverUrl string, modelName string, modelVersion string) TritonPredClient {
	// Connect to gRPC server
	conn, err := grpc.Dial(serverUrl, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		panic("fail to connect to triton server")
	}

	// Create client from gRPC server connection
	grpcClient := triton.NewGRPCInferenceServiceClient(conn)
	predClient := &tritonPredClient{
		grpcClient:   grpcClient,
		modelName:    modelName,
		modelVersion: modelVersion,
	}

	liveResponse := predClient.ServerLive()
	if !liveResponse.Live {
		panic("server not live")
	}
	readyRepsonse := predClient.ServerReady()
	if !readyRepsonse.Ready {
		panic("server not ready")
	}
	metadataResponse := predClient.ModelMetadata(modelName, modelVersion)
	log.Printf("get model metadata: %v", metadataResponse)

	return predClient
}

func (c *tritonPredClient) Preprocess(inputs []string) []byte {
	var inputBytes []byte
	for _, input := range inputs {
		strBytes := []byte(input)
		lenBytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(lenBytes, uint32(len(strBytes)))
		inputBytes = append(inputBytes, lenBytes...)
		inputBytes = append(inputBytes, strBytes...)
	}
	return inputBytes
}

func (c *tritonPredClient) Postprocess(response *triton.ModelInferResponse) []dto.ClsResult {
	var predicts []string
	var scores []float32
	for i, output := range response.Outputs {
		rawOutput := response.RawOutputContents[i]
		if output.Name == "predict" {
			offset := 0
			for offset < len(rawOutput) {
				length := int(binary.LittleEndian.Uint32(rawOutput[offset : offset+4]))
				offset += 4
				predicts = append(predicts, string(rawOutput[offset:offset+length]))
				offset += length
			}
		}
		if output.Name == "score" {
			s := make([]float32, output.Shape[0])
			buf := bytes.NewReader(rawOutput)
			binary.Read(buf, binary.LittleEndian, &s)
			scores = append(scores, s...)
		}
	}

	if predicts == nil || scores == nil || len(predicts) != len(scores) {
		panic("fail to post-process")
	}

	results := []dto.ClsResult{}
	for i := 0; i < len(predicts); i++ {
		result := dto.ClsResult{
			Predict: predicts[i],
			Score:   scores[i],
		}
		results = append(results, result)
	}
	return results
}

func (c *tritonPredClient) ServerLive() *triton.ServerLiveResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverLiveRequest := triton.ServerLiveRequest{}
	// Submit ServerLive request to server
	serverLiveResponse, err := c.grpcClient.ServerLive(ctx, &serverLiveRequest)
	if err != nil {
		panic("fail to get server live result")
	}
	return serverLiveResponse
}

func (c *tritonPredClient) ServerReady() *triton.ServerReadyResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverReadyRequest := triton.ServerReadyRequest{}
	// Submit ServerReady request to server
	serverReadyResponse, err := c.grpcClient.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		panic("fail to get server ready result")
	}
	return serverReadyResponse
}

func (c *tritonPredClient) ModelMetadata(modelName string, modelVersion string) *triton.ModelMetadataResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create status request for a given model
	modelMetadataRequest := triton.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	// Submit modelMetadata request to server
	modelMetadataResponse, err := c.grpcClient.ModelMetadata(ctx, &modelMetadataRequest)
	if err != nil {
		panic("fail to get model metadata result")
	}
	return modelMetadataResponse
}

func (c *tritonPredClient) ModelInfer(rawInput []byte, modelName string, modelVersion string) *triton.ModelInferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request input tensors
	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "text",
			Datatype: "BYTES",
			Shape:    []int64{-1},
		},
	}

	// Create request input output tensors
	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "predict",
		},
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "score",
		},
	}

	// Create inference request for specific model/version
	modelInferRequest := triton.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput)

	// Submit inference request to server
	modelInferResponse, err := c.grpcClient.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		panic("fail to get model infer result")
	}
	return modelInferResponse
}

func (c *tritonPredClient) GetPreds(texts []string) []dto.ClsResult {
	fmt.Println(texts)
	rawInputs := c.Preprocess(texts)
	inferResponse := c.ModelInfer(rawInputs, c.modelName, c.modelVersion)
	return c.Postprocess(inferResponse)
}

func (c *tritonPredClient) IsReady() bool {
	liveResponse := c.ServerLive()
	if !liveResponse.Live {
		return false
	}
	readyRepsonse := c.ServerReady()
	return readyRepsonse.Ready
}
