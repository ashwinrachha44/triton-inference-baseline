package dto

type ClsResult struct {
	Predict string  `json:"predict"`
	Score   float32 `json:"score"`
}

type SentResult struct {
	Text       string `json:"text"`
	SentIndex  int64  `json:"sent_index"`
	GroupIndex int64  `json:"group_index"`
}

type SentClsResult struct {
	Index   int64   `json:"index"`
	Text    string  `json:"text"`
	Predict string  `json:"predict"`
	Score   float32 `json:"score"`
}

type PredResult struct {
	SentPreds []SentClsResult `json:"sent_preds"`
}
