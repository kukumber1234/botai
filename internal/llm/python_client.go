package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

type PythonClient struct {
	URL string
}

type askRequest struct {
	Question string `json:"question"`
}

type askResponse struct {
	Answer string `json:"answer"`
}

func NewPythonClient(url string) *PythonClient {
	return &PythonClient{URL: url}
}

func (c *PythonClient) Ask(question string, _ string) (string, error) {
	payLoad := askRequest{Question: question}
	data, err := json.Marshal(payLoad)
	if err != nil {
		return "", err
	}

	resp, err := http.Post(c.URL+"/ask", "application/json", bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", errors.New("Python AI API returned status: " + resp.Status)
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	var result askResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return "", err
	}

	return result.Answer, nil
}
