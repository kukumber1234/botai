package telegram

import (
	"log"
	"strings"

	"CyberLawKZBot/internal/core/port"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

func StartBot(token string, responder port.Responder) error {
	bot, err := tgbotapi.NewBotAPI(token)
	if err != nil {
		return err
	}

	log.Printf("Бот запущен: %s", bot.Self.UserName)

	u := tgbotapi.NewUpdate(0)
	u.Timeout = 60
	updates := bot.GetUpdatesChan(u)

	for update := range updates {
		if update.Message == nil {
			continue
		}

		text := strings.TrimSpace(update.Message.Text)
		var reply string

		switch text {
		case "/start":
			reply = "Привет! Я бот по вопросам кибербезопасности РК."
		case "/help":
			reply = "Я могу отвечать на вопросы по законам РК в сфере ИБ."
		default:
			typing := tgbotapi.NewChatAction(update.Message.Chat.ID, tgbotapi.ChatTyping)
			_, _ = bot.Send(typing)

			reply, err = responder.Respond(text)
			if err != nil {
				log.Fatal("Ошибка при обращений к модели: ", err)
			}
		}

		reply = sanitizeReply(reply)

		msg := tgbotapi.NewMessage(update.Message.Chat.ID, reply)
		if _, err := bot.Send(msg); err != nil {
			log.Printf("Ошибка при отправке сообщения: %v", err)
		}
	}
	return nil
}

func sanitizeReply(s string) string {
	fallback := "Простите, я могу отвечать только на вопросы по кибербезопасности и законам РК."

	s = strings.TrimSpace(s)
	if s == "" {
		return fallback
	}
	return s
}
