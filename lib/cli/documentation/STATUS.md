# Project Status

## Completed Features:

- Core CLI tool (`guido`) for running local and cloud LLMs from a single binary
- Support for multiple backends: llama.cpp (local/embedded), OpenAI, Anthropic
- Three main interaction modes:
    - `complete`: one-shot prompt completion
    - `chat`: interactive multi-turn conversation
    - `serve`: persistent OpenAI-compatible HTTP server
- Built-in agentic tool-calling loop with web search and MCP (Model Context Protocol) support across all three modes
- Lazy-loading for local models — server starts instantly, model loads on first request and unloads after idle timeout
- Full streaming support across all backends and endpoints
- Multimodal/vision support via llama.cpp's mmproj
- Self contained binary with llama.cpp tools embedded via `go:embed`
- Structured logging system tracking every interaction with per-turn detail
- Resumable and renameable chat sessions persisted to local JSON files
- Create ability to switch models during a chat session and automatically migrate conversation history to new model's context window

## To Do:
- Change default log for chats / completions / etc to be date and time with nothing else. Keep all other things about logs, just the way we write the file name by default.
- Discover if we can monitor context window usage for chats
    - If possible, we need to set context limits on a per-model basis and warn users when they're approaching the limit
    - At a certain threshold, if possible, we need to be able to compact conversation history to free up context space while preserving as much relevant information as possible
- Update ability to set a model's default system prompt and have it persist across sessions from the config.yaml file
    - And, if possible, we can dynamically change the system prompt depending on the users input
- Update ability to set a model's temperature and have it persist across sessions from the config.yaml file
    - And, if possible, support the ability to update the current temperature in a chat session
- Update the ability to set a model's max user and response tokens in the config.yaml file
    - I think this is currently set when we start a chat session in the Go code and it should be model specific
- Implement the analyze feature for defined agentic workflow loops
- Create basic low level cli tools for converting hugging face models to ggufs and storing them in the correct directory for a user's system
- Add support for more backends (e.g. Azure, Google, Hugging Face)