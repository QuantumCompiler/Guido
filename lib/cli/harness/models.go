package harness

// ChatMessage represents a single message in a conversation
type ChatMessage struct {
	Role    string `json:"role"`    // "system", "user", or "assistant"
	Content string `json:"content"`
}

// ChatRequest represents a multi-turn chat request
type ChatRequest struct {
	Messages    []ChatMessage `json:"messages"`
	Model       string        `json:"model"`
	Temperature float32       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
	Stream      bool          `json:"stream"`
}

// ChatResponse represents the response from a chat request
type ChatResponse struct {
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
	TokensUsed   int         `json:"tokens_used"`
	Model        string      `json:"model"`
}

// CompletionRequest represents a request to complete a prompt
type CompletionRequest struct {
	Prompt      string  `json:"prompt"`
	Temperature float32 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
	StreamMode  bool    `json:"stream"`
	Model       string  `json:"model"`
}

// CompletionResponse represents the response from a completion request
type CompletionResponse struct {
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"` // "stop", "length", "error", etc.
	TokensUsed   int    `json:"tokens_used"`
	Model        string `json:"model"`
}

// ModelInfo contains metadata about an available model
type ModelInfo struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Provider string `json:"provider"` // "openai", "anthropic", "llamacpp"
	Type     string `json:"type"`     // "chat", "completion"
}

// Config represents the overall harness configuration
type Config struct {
	Server   ServerConfig            `yaml:"server" json:"server"`
	Models   ModelsConfig            `yaml:"models" json:"models"`
	Backends map[string]BackendConfig `yaml:"backends" json:"backends"`
}

// ServerConfig contains server-specific configuration
type ServerConfig struct {
	Port int    `yaml:"port" json:"port"`
	Mode string `yaml:"mode" json:"mode"` // "http" or "cli"
}

// ModelsConfig contains model-related configuration
type ModelsConfig struct {
	Default string `yaml:"default" json:"default"`
}

// BackendConfig represents configuration for a specific backend
type BackendConfig struct {
	Type         string                 `yaml:"type" json:"type"`                   // "llamacpp", "openai", "anthropic", "mock", "huggingface"
	APIKey       string                 `yaml:"api_key" json:"api_key"`
	URL          string                 `yaml:"url" json:"url"`
	Model        string                 `yaml:"model" json:"model"`
	ModelPath    string                 `yaml:"model_path" json:"model_path"`       // Path to local model file for embedded servers
	Port         int                    `yaml:"port" json:"port"`                   // Port for embedded servers (default auto-assigned)
	ChatTemplate string                 `yaml:"chat_template" json:"chat_template"` // e.g. "gemma", "llama3", "chatml", "mistral"
	GPULayers    int                    `yaml:"gpu_layers" json:"gpu_layers"`       // Override GPU layers (default 99)
	Extra        map[string]interface{} `yaml:"-" json:"-"`
}
