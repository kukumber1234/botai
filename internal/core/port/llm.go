package port

type LLMClient interface {
	Ask(question string, context string) (string, error)
}