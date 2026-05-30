package logger

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// Logger writes operational events to daily .log files under logDir/main/ and
// per-chat metrics to individual JSON files under logDir/chats/ (CLI) or
// logDir/http/ (HTTP server).
type Logger struct {
	mu             sync.Mutex
	mainDir        string // operational .log files
	chatsDir       string // CLI per-chat JSON files
	httpDir        string // HTTP per-chat JSON files
	completionsDir string // one-shot completion JSON files

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
		mainDir:        filepath.Join(logDir, "main"),
		chatsDir:       filepath.Join(logDir, "chats"),
		httpDir:        filepath.Join(logDir, "http"),
		completionsDir: filepath.Join(logDir, "completions"),
	}
	for _, dir := range []string{l.mainDir, l.chatsDir, l.httpDir, l.completionsDir} {
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

	now := time.Now().Local()
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
// TurnId is a 1-based sequential integer identifying the turn.
// UserTokens is the prompt token count for this turn; ModelTokens is the
// completion token count. The top-level ChatMetrics fields with the same names
// are the running totals across all turns.
type TurnRecord struct {
	TurnId       int      `json:"turnId"`
	UserInput    string   `json:"user_input"`
	ModelOutput  string   `json:"model_output"`
	UserTokens   int      `json:"userTokens"`
	ModelTokens  int      `json:"modelTokens"`
	InvokedTools []string `json:"invoked_tools"`
	SentAt       string   `json:"sent_at"`      // when the user prompt was sent
	RespondedAt  string   `json:"responded_at"` // when the model response completed
}

// ChatMetrics is the schema written to each per-chat JSON file. One file
// represents a single chat (identified by ChatID) and contains every turn.
type ChatMetrics struct {
	ChatID string `json:"chat_id"`
	// CustomName is a user-assigned name for the chat. It is empty by default,
	// in which case ChatID is used as the file name. When set (via RenameChat),
	// the file is named "<CustomName>.json" while ChatID still records the
	// original, system-generated name.
	CustomName     string   `json:"custom_name"`
	Model          string   `json:"model"`
	StartedAt      string   `json:"started_at"`
	EndedAt        string   `json:"ended_at,omitempty"`
	DurationMs     int64    `json:"duration_ms,omitempty"`
	AvailableTools []string `json:"available_tools"`
	Streaming      bool     `json:"streaming"`
	FinishReason   string   `json:"finish_reason,omitempty"`
	// Estimated is true when token counts were derived heuristically from text
	// rather than reported by the backend (streaming responses don't surface a
	// usage block, so counts are approximated at ~4 characters per token).
	Estimated bool `json:"estimated,omitempty"`
	// UserTokens and ModelTokens are running totals updated after every turn.
	UserTokens  int          `json:"userTokens"`
	ModelTokens int          `json:"modelTokens"`
	Turns       []TurnRecord `json:"turns"`
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

// NewCompletionSession starts tracking a one-shot completion (the `complete`
// command and the /v1/completions endpoint); the JSON file is written under
// logs/completions/. See newSession for parameter semantics.
func (l *Logger) NewCompletionSession(chatID, model string, availableTools []string, streaming bool) *ChatSession {
	return l.newSession(l.completionsDir, chatID, model, availableTools, streaming)
}

// newSession starts tracking a new chat written to dir. availableTools is the
// list of tools offered to the model (may be nil). streaming indicates whether
// the response was streamed to the client. The first turn's sent-time defaults
// to now; call BeginTurn before each subsequent turn to stamp its own sent-time.
func (l *Logger) newSession(dir, chatID, model string, availableTools []string, streaming bool) *ChatSession {
	if availableTools == nil {
		availableTools = []string{}
	}
	now := time.Now().Local()
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
	s.pendingSentAt = time.Now().Local()
}

// RecordTurn appends one user↔model exchange, updates the running token totals,
// emits a "model responded" log line, and persists the chat JSON.
// sent_at comes from the most recent BeginTurn (or chat start); responded_at is
// stamped now. userTokens is the prompt token count; modelTokens is the
// completion token count for this turn.
func (s *ChatSession) RecordTurn(userInput, modelOutput string, userTokens, modelTokens int, invokedTools []string) {
	now := time.Now().Local()
	sent := s.pendingSentAt
	if sent.IsZero() {
		sent = now
	}
	if invokedTools == nil {
		invokedTools = []string{}
	}

	turnNum := len(s.metrics.Turns) + 1 // 1-based sequential id

	s.metrics.Turns = append(s.metrics.Turns, TurnRecord{
		TurnId:       turnNum,
		UserInput:    userInput,
		ModelOutput:  modelOutput,
		UserTokens:   userTokens,
		ModelTokens:  modelTokens,
		InvokedTools: invokedTools,
		SentAt:       sent.Format(time.RFC3339Nano),
		RespondedAt:  now.Format(time.RFC3339Nano),
	})

	// Update running totals at the top level.
	s.metrics.UserTokens += userTokens
	s.metrics.ModelTokens += modelTokens

	s.pendingSentAt = time.Time{}

	s.logger.ModelResponded(
		s.metrics.ChatID, s.metrics.Model,
		userTokens, modelTokens,
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

// ─── Resuming previous chats ──────────────────────────────────────────────────

// LoadChat reads a previously saved CLI chat from logs/chats/. The name may be
// either the chat's current file name (its custom name if renamed) or its
// original chat id; the file name is tried first, then a scan by chat_id.
func (l *Logger) LoadChat(name string) (*ChatMetrics, error) {
	path := filepath.Join(l.chatsDir, name+".json")
	if data, err := os.ReadFile(path); err == nil {
		var m ChatMetrics
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, fmt.Errorf("parse chat %q: %w", name, err)
		}
		return &m, nil
	}

	// Fall back to a scan: the chat may have been renamed, so its file name no
	// longer matches its original chat id.
	chats, err := l.ListChats()
	if err != nil {
		return nil, fmt.Errorf("load chat %q: %w", name, err)
	}
	for i := range chats {
		if chats[i].ChatID == name {
			return &chats[i], nil
		}
	}
	return nil, fmt.Errorf("load chat %q: not found", name)
}

// RenameChat assigns a new custom name to a saved CLI chat. The chat is located
// by its current file name (its custom name if previously renamed, otherwise its
// chat id). The JSON file is renamed to "<newName>.json", the CustomName field
// is updated, and the original ChatID is preserved. Returns the updated metrics.
//
// newName is sanitized for filesystem safety. It is an error to rename to a name
// that already exists (to avoid clobbering another chat).
func (l *Logger) RenameChat(currentName, newName string) (*ChatMetrics, error) {
	newName = sanitizeForFilename(newName)
	if newName == "" {
		return nil, fmt.Errorf("rename chat: new name is empty after sanitization")
	}

	oldPath := filepath.Join(l.chatsDir, currentName+".json")
	data, err := os.ReadFile(oldPath)
	if err != nil {
		return nil, fmt.Errorf("rename chat %q: %w", currentName, err)
	}
	var m ChatMetrics
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("rename chat %q: parse: %w", currentName, err)
	}

	if newName == currentName {
		return &m, nil // no-op
	}

	newPath := filepath.Join(l.chatsDir, newName+".json")
	if _, err := os.Stat(newPath); err == nil {
		return nil, fmt.Errorf("rename chat: %q already exists", newName)
	}

	// Update the metadata, write to the new path, then remove the old file.
	m.CustomName = newName
	out, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("rename chat: marshal: %w", err)
	}
	if err := os.WriteFile(newPath, out, 0o644); err != nil {
		return nil, fmt.Errorf("rename chat: write %q: %w", newName, err)
	}
	if err := os.Remove(oldPath); err != nil {
		// New file is already written; best-effort cleanup of the old one.
		l.Error("rename chat: remove old file", err)
	}
	return &m, nil
}

// ListChats returns saved CLI chats from logs/chats/, sorted oldest-first by
// start time.
func (l *Logger) ListChats() ([]ChatMetrics, error) {
	entries, err := os.ReadDir(l.chatsDir)
	if err != nil {
		return nil, fmt.Errorf("list chats: %w", err)
	}
	var chats []ChatMetrics
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(l.chatsDir, e.Name()))
		if err != nil {
			continue
		}
		var m ChatMetrics
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		chats = append(chats, m)
	}
	sort.Slice(chats, func(i, j int) bool {
		return chats[i].StartedAt < chats[j].StartedAt
	})
	return chats, nil
}

// LatestChat returns the most recently started saved CLI chat, or an error if
// none exist.
func (l *Logger) LatestChat() (*ChatMetrics, error) {
	chats, err := l.ListChats()
	if err != nil {
		return nil, err
	}
	if len(chats) == 0 {
		return nil, fmt.Errorf("no saved chats found")
	}
	m := chats[len(chats)-1]
	return &m, nil
}

// ResumeChatSession reopens a previously saved chat so new turns are appended to
// the same JSON file. Prior turns, chat id, and original start time are
// preserved. availableTools and streaming reflect the resumed session.
func (l *Logger) ResumeChatSession(m *ChatMetrics, availableTools []string, streaming bool) *ChatSession {
	if availableTools == nil {
		availableTools = []string{}
	}
	now := time.Now().Local()
	started := now
	if t, err := time.Parse(time.RFC3339Nano, m.StartedAt); err == nil {
		started = t
	}
	metrics := *m
	metrics.AvailableTools = availableTools
	metrics.Streaming = streaming
	metrics.EndedAt = ""
	metrics.FinishReason = ""
	if metrics.Turns == nil {
		metrics.Turns = []TurnRecord{}
	}
	return &ChatSession{
		logger:        l,
		dir:           l.chatsDir,
		startedAt:     started,
		pendingSentAt: now,
		metrics:       metrics,
	}
}

// Finish records the end time and writes the final JSON file.
func (s *ChatSession) Finish(finishReason string) {
	now := time.Now().Local()
	s.metrics.EndedAt = now.Format(time.RFC3339Nano)
	s.metrics.DurationMs = now.Sub(s.startedAt).Milliseconds()
	s.metrics.FinishReason = finishReason
	s.persist()
}

// fileName returns the base file name (no extension) for a chat: the custom
// name if one is set, otherwise the original chat id.
func (m *ChatMetrics) fileName() string {
	if m.CustomName != "" {
		return m.CustomName
	}
	return m.ChatID
}

// persist writes the current metrics to the chat's JSON file. Called after each
// turn so the file stays current even if the process exits before Finish.
func (s *ChatSession) persist() {
	path := filepath.Join(s.dir, s.metrics.fileName()+".json")
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

// NewChatID generates a chat identifier derived from the model name and the
// current UTC time, suitable for use as a filename.
// Format: "<model>-MM-DD-YYYY:HH-MM-SS", e.g. "llama3.3-05-30-2026:14-23-09".
// The model name is sanitized so any character outside [A-Za-z0-9._-] is
// replaced with '-' (e.g. "org/model" becomes "org-model").
func NewChatID(model string) string {
	return fmt.Sprintf("%s-%s", sanitizeForFilename(model), time.Now().Local().Format("01-02-2006:15-04-05"))
}

// sanitizeForFilename replaces any character outside [A-Za-z0-9._-] with '-' so
// the result is safe to use as part of a filename.
func sanitizeForFilename(s string) string {
	return filenameUnsafe.ReplaceAllString(s, "-")
}

var filenameUnsafe = regexp.MustCompile(`[^A-Za-z0-9._-]`)

// EstimateTokens approximates the token count of s using a ~4-characters-per-token
// heuristic. Used for streaming responses where the backend doesn't report usage.
func EstimateTokens(s string) int {
	if s == "" {
		return 0
	}
	return (len(s) + 3) / 4
}
