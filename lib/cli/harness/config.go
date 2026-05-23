package harness

import (
	"fmt"
	"os"

	"github.com/spf13/viper"
)

// LoadConfig loads configuration from a file or environment
func LoadConfig(configPath string) (*Config, error) {
	v := viper.New()

	// Set defaults
	v.SetDefault("server.port", 8080)
	v.SetDefault("server.mode", "http")
	v.SetDefault("models.default", "openai")

	if configPath != "" {
		v.SetConfigFile(configPath)
		if err := v.ReadInConfig(); err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	// Bind environment variables
	v.BindEnv("backends.openai.api_key", "OPENAI_API_KEY")
	v.BindEnv("backends.anthropic.api_key", "ANTHROPIC_API_KEY")
	v.BindEnv("backends.llamacpp.url", "LLAMACPP_URL")

	// Allow env var overrides
	v.AutomaticEnv()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	if err := validateConfig(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// LoadConfigFromEnv loads configuration only from environment variables
func LoadConfigFromEnv() (*Config, error) {
	v := viper.New()

	// Set defaults
	v.SetDefault("server.port", 8080)
	v.SetDefault("server.mode", "http")
	v.SetDefault("models.default", "openai")

	// Bind environment variables
	v.BindEnv("backends.openai.api_key", "OPENAI_API_KEY")
	v.BindEnv("backends.anthropic.api_key", "ANTHROPIC_API_KEY")
	v.BindEnv("backends.llamacpp.url", "LLAMACPP_URL")

	// Allow env var overrides
	v.AutomaticEnv()

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	if err := validateConfig(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// validateConfig checks if the configuration is valid
func validateConfig(cfg *Config) error {
	if cfg.Server.Port < 1 || cfg.Server.Port > 65535 {
		return fmt.Errorf("%w: invalid port number %d", ErrInvalidConfig, cfg.Server.Port)
	}

	if cfg.Server.Mode != "http" && cfg.Server.Mode != "cli" {
		return fmt.Errorf("%w: server mode must be 'http' or 'cli', got %s", ErrInvalidConfig, cfg.Server.Mode)
	}

	if cfg.Models.Default == "" {
		return fmt.Errorf("%w: default model must be specified", ErrInvalidConfig)
	}

	return nil
}

// SaveConfigTemplate writes a sample config file to the given path
func SaveConfigTemplate(filePath string) error {
	template := `server:
  port: 8080
  mode: http              # 'http' for service, 'cli' for command-line

models:
  default: openai         # Default model to use if not specified

backends:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-opus"

  llamacpp:
    url: "http://localhost:8000"
    model: "llama-2"
`

	return os.WriteFile(filePath, []byte(template), 0644)
}
