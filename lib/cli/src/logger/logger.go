package logger

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Logger writes operational events to daily .log files under logDir/main/ and
// per-chat metrics to individual JSON files under logDir/chats/ (CLI) or
// logDir/http/ (HTTP server).
type Logger struct {
	mu       sync.Mutex
	mainDir  string // operational .log files
	chatsDir string // CLI per-chat JSON files
	httpDir  string // HTTP per-chat JSON files

	// current open log file and the date it was opened for
	file    *os.File
	fileDay string // "2006-01-02"
}

// New creates a Logger that writes under logDir. The main/, chats/, and http/
// subdirectories are created up front.
func New(logDir string) (*Logger, error) {
	if logDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}
		logDir = filepath.Join(home, ".guido", "logs")
	}
	l := &Logger{
		mainDir:  filepath.Join(logDir, "main"),
		chatsDir: filepath.Join(logDir, "chats"),
		httpDir:  filepath.Join(logDir, "http"),
	}
	for _, dir := range []string{l.mainDir, l.chatsDir, l.httpDir} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("logger: create dirs: %w", err)
		}
	}
	return l, nil
}

// ─── Operational log (.log) ───────────────────────────────────────────────────

func (l *Logger) write(level, msg string) {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := time.Now().UTC()
	day := now.Format("2006-01-02")

	// Rotate file on date change.
	if l.file == nil || l.fileDay != day {
		if l.file != nil {
			l.file.Close()
		}
		path := filepath.Join(l.mainDir, "guido-"+day+".log")
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return
		}
		l.file = f
		l.fileDay = day
	}

	fmt.Fprintf(l.file, "%s [%s] %s\n", now.Format(time.RFC3339), level, msg)
}

// ModelLoaded logs that a backend finished loading a model.
func (l *Logger) ModelLoaded(backend, model string) {
	l.write("INFO", fmt.Sprintf("model loaded: backend=%s model=%s", backend, model))
}

// ModelUnloaded logs that a backend unloaded a model (idle timeout).
func (l *Logger) ModelUnloaded(backend, model string) {
	l.write("INFO", fmt.Sprintf("model unloaded: backend=%s model=%s", backend, model))
}

// CompleteCall logs a single-turn completion request.
// from is "cli" or "http".
func (l *Logger) CompleteCall(model, from string) {
	l.write("INFO", fmt.Sprintf("complete called: model=%s from=%s", model, from))
}

// ChatSubmitted logs a user chat submission. The model name leads; the chat id
// and available tools are appended at the end of the line.
// tools may be nil/empty if no tools are active.
func (l *Logger) ChatSubmitted(chatID, model string, tools []string) {
	toolStr := "none"
	if len(tools) > 0 {
		toolStr = strings.Join(tools, ",")
	}
	l.write("INFO", fmt.Sprintf("user chat submitted to model=%s id=%s tools=[%s]", model, chatID, toolStr))
}

// ModelResponded logs a single model response (one turn), including the token
// counts for the user prompt and the generated output and the turn latency.
// The chat id is appended at the end of the line.
func (l *Logger) ModelResponded(chatID, model string, promptTokens, completionTokens int, latencyMs int64, estimated bool) {
	suffix := ""
	if estimated {
		suffix = " (estimated)"
	}
	l.write("INFO", fmt.Sprintf(
		"model=%s responded prompt_tokens=%d%s completion_tokens=%d%s latency=%dms id=%s",
		model, promptTokens, suffix, completionTokens, suffix, latencyMs, chatID,
	))
}

// ToolInvoked logs a single tool call within a chat.
func (l *Logger) ToolInvoked(chatID, toolName string) {
	l.write("INFO", fmt.Sprintf("tool invoked: id=%s tool=%s", chatID, toolName))
}

// Error logs an error event.
func (l *Logger) Error(context string, err error) {
	l.write("ERROR", fmt.Sprintf("%s: %v", context, err))
}

// Info logs a free-form informational message.
func (l *Logger) Info(msg string) {
	l.write("INFO", msg)
}

// Close flushes and closes the open log file.
func (l *Logger) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file != nil {
		l.file.Close()
		l.file = nil
	}
}

// ─── Per-chat JSON metrics ────────────────────────────────────────────────────

// TurnRecord captures a single user↔model exchange within a chat.
type TurnRecord struct {
	UserInput        string   `json:"user_input"`
	ModelOutput      string   `json:"model_output"`
	PromptTokens     int      `json:"prompt_tokens"`
	CompletionTokens int      `json:"completion_tokens"`
	TotalTokens      int      `json:"total_tokens"`
	InvokedTools     []string `json:"invoked_tools"`
	SentAt           string   `json:"sent_at"`      // when the user prompt was sent
	RespondedAt      string   `json:"responded_at"` // when the model response completed
}

// ChatMetrics is the schema written to each per-chat JSON file. One file
// represents a single chat (identified by ChatID) and contains every turn.
type ChatMetrics struct {
	ChatID         string       `json:"chat_id"`
	Model          string       `json:"model"`
	StartedAt      string       `json:"started_at"`
	EndedAt        string       `json:"ended_at,omitempty"`
	DurationMs     int64        `json:"duration_ms,omitempty"`
	AvailableTools []string     `json:"available_tools"`
	Streaming      bool         `json:"streaming"`
	FinishReason   string       `json:"finish_reason,omitempty"`
	// Estimated is true when token counts were derived heuristically from text
	// rather than reported by the backend (streaming responses don't surface a
	// usage block, so counts are approximated at ~4 characters per token).
	Estimated bool         `json:"estimated,omitempty"`
	Turns     []TurnRecord `json:"turns"`
}

// ChatSession accumulates metrics for one chat and writes them to JSON on Finish.
type ChatSession struct {
	logger        *Logger
	dir           string // directory the JSON file is written to
	metrics       ChatMetrics
	startedAt     time.Time
	pendingSentAt time.Time // timestamp the current turn's prompt was sent
}

// NewChatSession starts tracking a CLI chat; the JSON file is written under
// logs/chats/. See newSession for parameter semantics.
func (l *Logger) NewChatSession(chatID, model string, availableTools []string, streaming bool) *ChatSession {
	return l.newSession(l.chatsDir, chatID, model, availableTools, streaming)
}

// NewHTTPSession starts tracking an HTTP-server chat; the JSON file is written
// under logs/http/. See newSession for parameter semantics.
func (l *Logger) NewHTTPSession(chatID, model string, availableTools []string, streaming bool) *ChatSession {
	return l.newSession(l.httpDir, chatID, model, availableTools, streaming)
}

// newSession starts tracking a new chat written to dir. availableTools is the
// list of tools offered to the model (may be nil). streaming indicates whether
// the response was streamed to the client. The first turn's sent-time defaults
// to now; call BeginTurn before each subsequent turn to stamp its own sent-time.
func (l *Logger) newSession(dir, chatID, model string, availableTools []string, streaming bool) *ChatSession {
	if availableTools == nil {
		availableTools = []string{}
	}
	now := time.Now().UTC()
	return &ChatSession{
		logger:        l,
		dir:           dir,
		startedAt:     now,
		pendingSentAt: now,
		metrics: ChatMetrics{
			ChatID:         chatID,
			Model:          model,
			StartedAt:      now.Format(time.RFC3339Nano),
			AvailableTools: availableTools,
			Streaming:      streaming,
			Turns:          []TurnRecord{},
		},
	}
}

// BeginTurn stamps the moment a user prompt is sent. Call it right before
// dispatching a turn so the recorded sent_at reflects that turn (not the chat
// start). The first turn is stamped automatically by NewChatSession.
func (s *ChatSession) BeginTurn() {
	s.pendingSentAt = time.Now().UTC()
}

// RecordTurn appends one user↔model exchange, emits a "model responded" log
// line for it, and persists the chat JSON. sent_at comes from the most recent
// BeginTurn (or chat start); responded_at is stamped now.
func (s *ChatSession) RecordTurn(userInput, modelOutput string, promptTokens, completionTokens int, invokedTools []string) {
	now := time.Now().UTC()
	sent := s.pendingSentAt
	if sent.IsZero() {
		sent = now
	}
	if invokedTools == nil {
		invokedTools = []string{}
	}
	s.metrics.Turns = append(s.metrics.Turns, TurnRecord{
		UserInput:        userInput,
		ModelOutput:      modelOutput,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
		InvokedTools:     invokedTools,
		SentAt:           sent.Format(time.RFC3339Nano),
		RespondedAt:      now.Format(time.RFC3339Nano),
	})
	s.pendingSentAt = time.Time{}

	s.logger.ModelResponded(
		s.metrics.ChatID, s.metrics.Model,
		promptTokens, completionTokens,
		now.Sub(sent).Milliseconds(), s.metrics.Estimated,
	)
	s.persist()
}

// MarkEstimated flags this session's token counts as heuristic estimates.
func (s *ChatSession) MarkEstimated() {
	s.metrics.Estimated = true
}

// ChatID returns this session's chat identifier.
func (s *ChatSession) ChatID() string {
	return s.metrics.ChatID
}

// Finish records the end time and writes the final JSON file.
func (s *ChatSession) Finish(finishReason string) {
	now := time.Now().UTC()
	s.metrics.EndedAt = now.Format(time.RFC3339Nano)
	s.metrics.DurationMs = now.Sub(s.startedAt).Milliseconds()
	s.metrics.FinishReason = finishReason
	s.persist()
}

// persist writes the current metrics to the chat's JSON file. Called after each
// turn so the file stays current even if the process exits before Finish.
func (s *ChatSession) persist() {
	path := filepath.Join(s.dir, s.metrics.ChatID+".json")
	f, err := os.Create(path)
	if err != nil {
		s.logger.Error("chat session write", err)
		return
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.Encode(s.metrics)
}

// ─── Chat ID generation ───────────────────────────────────────────────────────

// NewChatID generates a unique chat identifier of the form
// "chat-<timestamp>-<hex>" suitable for filenames.
func NewChatID() string {
	return fmt.Sprintf("chat-%d-%04x", time.Now().UnixNano(), rand.Intn(0xffff))
}

// EstimateTokens approximates the token count of s using a ~4-characters-per-token
// heuristic. Used for streaming responses where the backend doesn't report usage.
func EstimateTokens(s string) int {
	if s == "" {
		return 0
	}
	return (len(s) + 3) / 4
}
