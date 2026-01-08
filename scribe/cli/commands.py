import json
import os
import subprocess
import sys
import tempfile
import uuid
import click
from rich.console import Console
from scribe.cli.constants import DEFAULT_PROVIDER
from pathlib import Path

from scribe.providers._provider_utils import get_provider
from scribe.providers.claude import CLAUDE_COPILOT_SETTINGS
from scribe.providers.gemini import get_copilot_settings as get_gemini_copilot_settings
from scribe.providers.codex import get_copilot_settings as get_codex_copilot_settings
from scribe.cli._cli_utils import get_python_path, merge_settings_intelligently


console = Console()


def copilot_impl(args=None, provider_name: str = DEFAULT_PROVIDER, verbose=False):
    """Implementation of copilot command - supports multiple AI providers."""
    provider = get_provider(provider_name)
    provider_display = provider.get_provider_display_name()

    # Give terminal a nice title for Gemini CLI
    if provider_name == "gemini":
        os.environ["CLI_TITLE"] = "Scribe - Copilot"

    # Generate session ID
    session_id = str(uuid.uuid4())
    os.environ["SCRIBE_SESSION_ID"] = session_id

    try:
        click.echo(f"\n🚀 Launching {provider_display} with Scribe notebook tools enabled...\n")
        python_path = get_python_path()

        if provider_name == "claude":
            # Create a dedicated subdirectory for scribe config files in the user's
            # home directory. We avoid the system temp directory because Claude Code's
            # file watcher may scan the temp directory and fail on socket files
            # (e.g., vscode-git-*.sock) with EOPNOTSUPP errors on macOS.
            # See: https://github.com/anthropics/claude-code/issues/14438
            # See: https://github.com/anthropics/claude-code/issues/15112
            scribe_config_dir = Path.home() / ".scribe" / "sessions"
            scribe_config_dir.mkdir(parents=True, exist_ok=True)

            # Create config files in the isolated scribe directory
            settings_file = scribe_config_dir / f"settings-{session_id}.json"
            with open(settings_file, "w") as f:
                json.dump(CLAUDE_COPILOT_SETTINGS, f)

            mcp_config_file = scribe_config_dir / f"mcp-config-{session_id}.json"
            with open(mcp_config_file, "w") as f:
                json.dump(provider.get_copilot_mcp_config(python_path), f)

            cmd = [
                "claude",
                "--mcp-config",
                str(mcp_config_file),
                "--settings",
                str(settings_file),
                "--append-system-prompt",
                "",
            ]
            # Add any additional args passed to scribe copilot
            if args:
                cmd.extend(args)
            try:
                subprocess.run(cmd)
            finally:
                # Clean up session-specific config files
                settings_file.unlink(missing_ok=True)
                mcp_config_file.unlink(missing_ok=True)
        elif provider_name == "gemini":
            # Get the settings from the provider
            settings_content = get_gemini_copilot_settings(python_path)

            # Create .gemini directory if it doesn't exist
            gemini_dir = Path(".gemini")
            gemini_dir.mkdir(exist_ok=True)

            settings_file = gemini_dir / "settings.json"

            # If settings file exists, merge intelligently
            if settings_file.exists():
                try:
                    with open(settings_file, "r") as f:
                        existing_settings = json.load(f)

                    settings_content = merge_settings_intelligently(
                        settings_content, existing_settings
                    )
                except (json.JSONDecodeError, IOError):
                    # If we can't read existing settings, just use new ones
                    pass

            # Write the settings file
            with open(settings_file, "w") as f:
                json.dump(settings_content, f, indent=2)

            # Build and run the gemini command using provider base (supports npx fallback)
            cmd = provider.get_command_base()

            # Add any additional args passed to scribe copilot
            if args:
                cmd.extend(args)

            subprocess.run(cmd)
        elif provider_name == "codex":
            # Check if we have piped input (non-interactive mode)
            import select
            has_input = select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
            
            if has_input:
                # Non-interactive mode: read stdin and use codex exec
                input_text = sys.stdin.read().strip()
                if input_text:
                    cmd = provider.get_command_base()
                    cmd.extend(["exec", input_text])
                    
                    # Add any additional args passed to scribe copilot
                    if args:
                        cmd.extend(args)
                    
                    subprocess.run(cmd)
                    return
            
            # Interactive mode: Build the codex command with -c overrides for MCP
            settings_content = get_codex_copilot_settings(python_path)

            # Extract scribe MCP config
            scribe_cfg = settings_content.get("mcp_servers", {}).get("scribe", {})
            mcp_command = scribe_cfg.get("command", python_path)
            mcp_args = scribe_cfg.get("args", [])
            mcp_env = scribe_cfg.get("env", {})

            cmd = provider.get_command_base()

            import json as _json

            # Command and args
            cmd.extend(["-c", f"mcp_servers.scribe.command={mcp_command}"])
            cmd.extend(["-c", f"mcp_servers.scribe.args={_json.dumps(mcp_args)}"])

            # Optional env passthroughs (like NOTEBOOK_OUTPUT_DIR)
            for _k, _v in mcp_env.items():
                cmd.extend(["-c", f"mcp_servers.scribe.env.{_k}={_v}"])

            # Add any additional args passed to scribe copilot
            if args:
                cmd.extend(args)

            subprocess.run(cmd)

        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    except FileNotFoundError:
        click.echo(f"Error: {provider_display} command not found.")
        click.echo(
            f"Please ensure {provider.get_provider_name()} CLI is installed and in PATH."
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error launching {provider_display}: {e}")
        sys.exit(1)


