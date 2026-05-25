package harness

import "errors"

var (
	// ErrNoAvailableBackend is returned when no backend can handle the requested model
	ErrNoAvailableBackend = errors.New("no available backend for the requested model")

	// ErrModelNotFound is returned when a model cannot be found
	ErrModelNotFound = errors.New("model not found")

	// ErrInvalidConfig is returned when configuration is invalid
	ErrInvalidConfig = errors.New("invalid configuration")

	// ErrProviderNotRegistered is returned when a provider is not registered
	ErrProviderNotRegistered = errors.New("provider not registered")
)
