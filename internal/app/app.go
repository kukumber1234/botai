package app

import (
	"CyberLawKZBot/internal/adapter/telegram"
	"CyberLawKZBot/internal/core/service"
	"CyberLawKZBot/internal/llm"
	"log"
	"os"

	"github.com/joho/godotenv"
)

func Start() {
	_ = godotenv.Load()

	token := os.Getenv("TELEGRAM_TOKEN")
	if token == "" {
		log.Fatal("TELEGRAM_TOKEN not set")
	}

	aiURL := os.Getenv("PYTHON_AI_URL")
	if aiURL == "" {
		log.Fatal("PYTHON_AI_URL not set")
	}
	llmClient := llm.NewPythonClient(aiURL)

	// model := os.Getenv("OLLAMA_MODEL")
	// if model == "" {
	// 	log.Fatal("OLLAMA_MODEL not set")
	// }

	// llmClient := llm.NewOllama(model)
	responder := service.NewResponder(llmClient)

	if err := telegram.StartBot(token, responder); err != nil {
		log.Fatal(err)
	}
}
