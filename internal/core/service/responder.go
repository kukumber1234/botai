package service

import (
	"CyberLawKZBot/internal/core/port"
)

type LLMResponder struct {
	llm port.LLMClient
}

func NewResponder(llm port.LLMClient) port.Responder {
	return &LLMResponder{llm: llm}
}

func (s *LLMResponder) Respond(question string) (string, error) {
	answer, err := s.llm.Ask(question, "")
	if err != nil {
		return "", err
	}
	return answer, nil
}
