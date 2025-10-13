# Scribe — Jupyter Server + Notebooks for CLI Agents
Give Claude Code, Codex, and Gemini CLI agents access to Jupyter servers + notebooks.

## Installation

```bash
# Install from GitHub using uv
uv add git+https://github.com/goodfire-ai/scribe.git
```

### Development Installation

For local development with editable install:

```bash
# Clone the repository
git clone https://github.com/goodfire-ai/scribe.git

# From your project directory, install scribe in editable mode
uv add --editable /path/to/scribe
```

## Usage  
Once installed, you can run the `scribe` command from within your virtual environment.  

This command will launch a CLI agent (the default is Claude Code, but you can update `DEFAULT_PROVIDER` in [constants.py](scribe/cli/constants.py)) with a notebook MCP server automatically enabled. Behind the scenes, a Jupyter server has been started and the agent has tools to run code that will be executed in an IPython kernel. The scribe server sits in between the agent and the Jupyter kernel, passing input code to the kernel and automatically writing all input code + kernel outputs (text, errors, images, etc.) to a Jupyter notebook.  

To specify a particular CLI agent, use `scribe claude`, `scribe codex`, or `scribe gemini`. These commands wrap calls to the underlying CLI agents — they will use your default auth method and other configurations, and you can pass CLI flags (e.g. `scribe claude -c` to continue a session).    


#### Start a new session
Once you've launched the CLI agent, ask it to start a new session. This will create a `notebooks/` directory wherever you launched the `scribe` command from, and will create a notebook with the current timestamp and a name provided by the agent.  
```
You: Start a new session for us to run some experiments on GPT-2.

Agent: I'll start a new Scribe session for image generation. [Tool call]

Agent: Session started successfully! I've created a new notebook at notebooks/2025-01-09-10-30_GPT-2_Experiments.ipynb. Where should we begin?
```

## Automatic MCP Permissions
**Claude Code**  
When running `scribe claude`, Claude Code is launched with a command-line argument that enables the MCP server and automatically enables most specific tool calls (e.g. starting a new session and executing code).  

**Codex**  
When running `scribe codex`, Codex is launched with a command-line argument that enables the MCP server using the `--config` flag [documented here](https://github.com/openai/codex/blob/main/docs/config.md).  

**Gemini CLI**  
When running `scribe gemini`, a `.gemini/settings.json` file is created (or updated if one already exists) with settings prepopulated to enable the MCP server with tool calls automatically enabled.  

## Security Note  
Agents can execute code via the Jupyter kernel that bypasses default CLI permissions. Use with caution.  
