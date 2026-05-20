APP     := bin/runner
VERSION := 0.1.0
BUILD   := $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")
LDFLAGS := -ldflags "-X main.version=$(VERSION) -X main.build=$(BUILD)"

.PHONY: build run clean install

build:
	go build $(LDFLAGS) -o $(APP) .

run:
	go run . $(ARGS)

install:
	go install $(LDFLAGS) .

clean:
	rm -f $(APP)