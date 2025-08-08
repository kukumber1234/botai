package port

type Responder interface {
	Respond(question string) (string, error)
} 