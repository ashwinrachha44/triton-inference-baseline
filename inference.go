package main

import (
	"triton-inference-baseline/client"
	"triton-inference-baseline/dto"
)

type ClassifierService interface {
	GetPredsPerSent(texts []string) []dto.PredResult
	GetPreds(texts []string) []dto.ClsResult
	IsReady() bool
}

type classifierService struct {
	sentClient client.TritonSentClient
	predClient client.TritonPredClient
}

func CreateClassifierService(sentClient client.TritonSentClient, predClient client.TritonPredClient) ClassifierService {
	return &classifierService{
		sentClient: sentClient,
		predClient: predClient,
	}
}

func (s *classifierService) GetPredsPerSent(texts []string) []dto.PredResult {
	sents := s.sentClient.GetSents(texts)
	sentTexts := []string{}
	for _, sent := range sents {
		sentTexts = append(sentTexts, sent.Text)
	}
	sentPreds := s.predClient.GetPreds(sentTexts)
	if len(sents) != len(sentPreds) {
		panic("sentence and result size is not equal")
	}
	var predResults []dto.PredResult
	for i := 0; i < len(texts); i++ {
		predResult := dto.PredResult{
			SentPreds: []dto.SentClsResult{},
		}
		predResults = append(predResults, predResult)
	}
	for i := 0; i < len(sents); i++ {
		sentResult := dto.SentClsResult{
			Index:   sents[i].SentIndex,
			Text:    sents[i].Text,
			Predict: sentPreds[i].Predict,
			Score:   sentPreds[i].Score,
		}
		j := int(sents[i].GroupIndex)
		predResults[j].SentPreds = append(predResults[j].SentPreds, sentResult)
	}
	return predResults
}

func (s *classifierService) GetPreds(texts []string) []dto.ClsResult {
	return s.predClient.GetPreds(texts)
}

func (s *classifierService) IsReady() bool {
	return s.predClient.IsReady()
}
