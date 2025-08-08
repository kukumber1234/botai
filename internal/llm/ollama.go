package llm

// import (
// 	domain "CyberLawKZBot/internal/core/domain/llm"
// 	"bytes"
// 	"encoding/json"
// 	"errors"
// 	"io"
// 	"net/http"
// 	"os"
// )

// type OllamaClient struct {
// 	model string
// }

// func NewOllama(model string) *OllamaClient {
// 	return &OllamaClient{model: model}
// }

// func (c *OllamaClient) Ask(question string, _ string) (string, error) {
// 	url := os.Getenv("URL_OLLAMA")

// 	reqBody := domain.OllamaRequest{
// 		Model:  c.model,
// 		Prompt: question,
// 		Stream: false,
// 	}

// 	data, err := json.Marshal(reqBody)
// 	if err != nil {
// 		return "", err
// 	}

// 	resp, err := http.Post(url, "application/json", bytes.NewBuffer(data))
// 	if err != nil {
// 		return "", err
// 	}
// 	defer resp.Body.Close()

// 	if resp.StatusCode != http.StatusOK {
// 		return "", errors.New("Ollama API returned status: " + resp.Status)
// 	}

// 	bodyBytes, err := io.ReadAll(resp.Body)
// 	if err != nil {
// 		return "", err
// 	}

// 	var result domain.OllamaResponse
// 	if err := json.Unmarshal(bodyBytes, &result); err != nil {
// 		return "", err
// 	}

// 	return result.Response, nil
// }
