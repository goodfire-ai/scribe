"""
Scribe Notebook MCP Server - Model Context Protocol interface for agents to work with
the Scribe notebook server. MCP endpoints are easier for agents to interact with than
raw HTTP requests or Jupyter Server API calls.
"""

import atexit
import os
import secrets
import signal
import subprocess
import sys
from typing import Dict, Any, Optional, List, Union

import requests
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from scribe.notebook._notebook_server_utils import (
    find_safe_port,
    check_server_health,
    start_scribe_server,
    cleanup_scribe_server,
    process_jupyter_outputs,
)  # noqa: E402


# Initialize MCP server
mcp = FastMCP("scribe")

# Global server management
_server_process: Optional[subprocess.Popen] = None
_server_port: Optional[int] = None
_server_url: Optional[str] = None
_server_token: Optional[str] = None
# Down the line, we may wish to keep the Jupyter server around even after MCP server exits
_is_external_server: bool = False

SCRIBE_PROVIDER: str = os.environ.get("SCRIBE_PROVIDER")

# Session tracking for cleanup
_active_sessions: set = set()


def start_jupyter_server() -> tuple[subprocess.Popen, int, str]:
    """Start a Jupyter server subprocess and return process, port, and URL."""
    port = find_safe_port()
    if port is None:
        raise Exception("Could not find an available port for Jupyter server")

    # Generate token for this server instance
    token = get_token()

    # Get notebook output directory from environment variable
    notebook_output_dir = os.environ.get("NOTEBOOK_OUTPUT_DIR")

    # Use utils function to start server
    process = start_scribe_server(port, token, notebook_output_dir)
    url = f"http://127.0.0.1:{port}"

    return process, port, url


def cleanup_server():
    """Clean up the managed Jupyter server."""
    global _server_process, _server_token, _active_sessions

    if _server_process and not _is_external_server:
        cleanup_scribe_server(_server_process)
        _server_process = None
        _server_token = None  # Clear token on cleanup


def ensure_server_running() -> str:
    """Ensure a Jupyter server is running and return its URL."""
    global _server_process, _server_port, _server_url, _is_external_server

    # Check if SCRIBE_PORT is set (external server)
    if "SCRIBE_PORT" in os.environ:
        port = os.environ["SCRIBE_PORT"]
        _server_port = int(port)
        _server_url = f"http://127.0.0.1:{port}"
        _is_external_server = True
        return _server_url

    # Check if our managed server is still running
    if _server_process and _server_process.poll() is None:
        return _server_url

    # Start a new managed server
    _is_external_server = False
    _server_process, _server_port, _server_url = start_jupyter_server()

    # Register cleanup handlers
    atexit.register(cleanup_server)
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_server())
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_server())

    print(f"Started managed Jupyter server at {_server_url}", file=sys.stderr)
    return _server_url


def get_token() -> str:
    """Generate or return cached auth token."""
    global _server_token
    if not _server_token and not _is_external_server:
        _server_token = secrets.token_urlsafe(32)
    return _server_token or ""


def get_server_status() -> Dict[str, Any]:
    """Get current server status information."""
    global _server_port, _server_url, _is_external_server, _server_process

    if not _server_url:
        return {
            "status": "not_started",
            "url": None,
            "port": None,
            "vscode_url": None,
            "will_shutdown_on_exit": True,
            "is_external": False,
            "health": "unknown",
        }

    # Check health using utils function
    health_data = check_server_health(_server_port) if _server_port else None
    health = "healthy" if health_data else "unreachable"

    # Check if process is still running (for managed servers)
    process_running = True
    if not _is_external_server and _server_process:
        process_running = _server_process.poll() is None

    return {
        "status": "running" if process_running else "stopped",
        "url": _server_url,
        "port": _server_port,
        "vscode_url": f"{_server_url}/?token={get_token()}" if _server_url else None,
        "will_shutdown_on_exit": not _is_external_server,
        "is_external": _is_external_server,
        "health": health,
    }


async def _start_session_internal(
    experiment_name: Optional[str] = None,
    notebook_path: Optional[str] = None,
    fork_prev_notebook: bool = True,
    tool_name: str = "start_session",
    kernel_name: str = "python3",
) -> Dict[str, Any]:
    """
    Internal helper function for starting sessions from scratch versus resuming versus forking existing notebook.

    Args:
        experiment_name: Custom name for the notebook
        notebook_path: Path to existing notebook (if any)
        fork_prev_notebook: If True, create new notebook; if False, use existing in-place
        tool_name: Name of the calling tool for logging/debugging
        kernel_name: Jupyter kernel spec name (e.g. "python3", "ir")
    """
    try:
        # Ensure server is running
        server_url = ensure_server_running()

        # Build request body
        request_body = {"kernel_name": kernel_name}
        if experiment_name:
            request_body["experiment_name"] = experiment_name
        if notebook_path:
            request_body["notebook_path"] = notebook_path
            request_body["fork_prev_notebook"] = fork_prev_notebook

        # Start session
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}
        print(f"[DEBUG MCP] {tool_name}: Connecting to {server_url}", file=sys.stderr)

        response = requests.post(
            f"{server_url}/api/scribe/start", json=request_body, headers=headers
        )
        response.raise_for_status()
        data = response.json()

        result = {
            "session_id": data["session_id"],
            "kernel_id": data.get("kernel_id"),
            "status": "started",
            "notebook_path": data["notebook_path"],
            "vscode_url": f"{data.get('server_url', server_url)}/?token={data.get('token', token)}",
            "kernel_name": data.get("kernel_name", "python3"),
            "kernel_display_name": data.get("kernel_display_name", "Scribe Kernel"),
        }

        # Track session for cleanup
        global _active_sessions
        _active_sessions.add(data["session_id"])

        # Handle restoration results if present (only for notebook-based sessions)
        if notebook_path:
            # Pass through restoration summary if present
            if "restoration_summary" in data:
                result["restoration_summary"] = data["restoration_summary"]

            # Only pass error details for debugging, not full restoration results
            if "restoration_results" in data:
                errors = [
                    r for r in data["restoration_results"] if r.get("status") == "error"
                ]
                if errors:
                    # Summarize errors with cell numbers and error messages only
                    error_summary = []
                    for error in errors:
                        error_info = {
                            "cell": error.get("cell"),
                            "error": error.get("error", "").split(":")[0]
                            if ":" in error.get("error", "")
                            else error.get("error", ""),
                        }
                        error_summary.append(error_info)
                    result["restoration_errors"] = error_summary

            # Add guidance for agent when working with existing notebooks
            if "restoration_summary" in data:
                has_errors = (
                    "restoration_errors" in result and result["restoration_errors"]
                )
                if fork_prev_notebook:
                    # Continue/fork scenario
                    if has_errors:
                        result["note"] = (
                            f"A new notebook has been created at {data['notebook_path']} "
                            f"with the restored state from {notebook_path}. "
                            f"Some cells had errors during restoration - see restoration_errors for details. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                    else:
                        result["note"] = (
                            f"A new notebook has been created at {data['notebook_path']} "
                            f"with the restored state from {notebook_path}. "
                            "All cells executed successfully during restoration. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                else:
                    # Resume scenario
                    if has_errors:
                        result["note"] = (
                            f"Resumed notebook at {data['notebook_path']} in-place. "
                            f"Some cells had errors during restoration - see restoration_errors for details. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                    else:
                        result["note"] = (
                            f"Successfully resumed notebook at {data['notebook_path']} in-place. "
                            "All cells executed successfully during restoration. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )

        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to start session ({tool_name}): {str(e)}")


@mcp.tool
async def start_new_session(
    experiment_name: Optional[str] = None,
    kernel_name: str = "python3",
) -> Dict[str, Any]:
    """
    Start a completely new Jupyter kernel session with an empty notebook.

    Supports any installed Jupyter kernel (e.g. "python3", "ir" for R via IRkernel).

    Args:
        experiment_name: Custom name for the notebook (e.g., "ImageGeneration")
        kernel_name: Jupyter kernel spec name. Defaults to "python3".

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the new notebook
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
    """
    return await _start_session_internal(
        experiment_name=experiment_name,
        notebook_path=None,
        fork_prev_notebook=True,
        tool_name="start_new_session",
        kernel_name=kernel_name,
    )


@mcp.tool(
    name="start_session_resume_notebook",
    # Description pulled from docstring
    tags=None,
    annotations={
        "title": "Start Session - Resume Notebook"  # A human-readable title for the tool.
    },
)
async def start_session_resume_notebook(
    notebook_path: str, kernel_name: str = "python3",
) -> Dict[str, Any]:
    """
    Start a new session by resuming an existing notebook in-place, modifying the original notebook file.

    This executes all cells in the existing notebook to restore the kernel state and updates
    the notebook file with new outputs. Use this to continue working in an existing notebook file.

    Args:
        notebook_path: Path to the existing notebook to resume from
        kernel_name: Jupyter kernel spec name. Should match the notebook's language.

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the resumed notebook (same as input)
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
        - restoration_summary: Summary of the resume operation
        - restoration_errors: List of any errors that occurred during cell execution
        - note: Guidance message about the resumed notebook
    """
    return await _start_session_internal(
        experiment_name=None,
        notebook_path=notebook_path,
        fork_prev_notebook=False,
        tool_name="start_session_resume_notebook",
        kernel_name=kernel_name,
    )


@mcp.tool
async def start_session_continue_notebook(
    notebook_path: str, experiment_name: Optional[str] = None,
    kernel_name: str = "python3",
) -> Dict[str, Any]:
    """
    Start a session by continuing from an existing notebook (creates a new notebook file).

    This creates a new notebook with "_continued" suffix, copies all cells from the existing
    notebook, and executes them to restore the kernel state. The original notebook is unchanged.

    Args:
        notebook_path: Path to the existing notebook to continue from
        experiment_name: Optional custom name for the new notebook
        kernel_name: Jupyter kernel spec name. Should match the notebook's language.

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the new notebook (with "_continued" suffix)
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
        - restoration_summary: Summary of the continuation operation
        - restoration_errors: List of any errors that occurred during cell execution
        - note: Guidance to read the new notebook
    """
    return await _start_session_internal(
        experiment_name=experiment_name,
        notebook_path=notebook_path,
        fork_prev_notebook=True,
        tool_name="start_session_continue_notebook",
        kernel_name=kernel_name,
    )


@mcp.tool
async def execute_code(
    session_id: str, code: str
) -> List[Union[Dict[str, Any], Image]]:
    """
    Execute Python code in the specified kernel session.

    Images generated during execution (e.g., via .show()) are returned as
    fastmcp.Image objects that can be directly viewed.

    Args:
        session_id: The session ID returned by start_session
        code: Python code to execute

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - execution_count: The cell execution number
        - outputs: List of output objects with type and content/data
    """
    # start_time = time.time()
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        # Process outputs using utils function
        outputs, images = process_jupyter_outputs(
            data["outputs"],
            session_id=session_id,
            save_images_locally=False,
        )


        # Create result list with execution metadata first, then images
        result = [
            {
                "session_id": session_id,
                "execution_count": data["execution_count"],
                "outputs": outputs,
            }
        ]
        result.extend(images)
        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to execute code: {str(e)}")


@mcp.tool
async def add_markdown(session_id: str, content: str) -> Dict[str, int]:
    """
    Add a markdown cell to the notebook for documentation.

    Args:
        session_id: The session ID
        content: Markdown content to add

    Returns:
        Dictionary with the cell number
    """
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/markdown",
            json={"session_id": session_id, "content": content},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()


        return {"cell_number": data["cell_number"]}

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to add markdown: {str(e)}")


@mcp.tool
async def edit_cell(
    session_id: str, code: str, cell_index: int = -1
) -> List[Union[Dict[str, Any], Image]]:
    """
    Edit an existing code cell in the notebook and execute the new code.

    This is especially useful for fixing errors or modifying the most recent cell.

    Args:
        session_id: The session ID
        code: New Python code to replace the cell content
        cell_index: Index of the code cell to edit (default -1 for last cell)
                   Use -1 for the most recent cell, -2 for second to last, etc.
                   Or use 0, 1, 2... for specific cells from the beginning

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - cell_index: The code cell index that was edited
        - actual_notebook_index: The actual index in the notebook (including markdown cells)
        - execution_count: The cell execution number
        - outputs: List of output objects with type and content/data
    """
    # start_time = time.time()
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": code, "cell_index": cell_index},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        # Process outputs using utils function
        outputs, images = process_jupyter_outputs(
            data["outputs"],
            session_id=session_id,
            save_images_locally=False,
        )


        # Create result list with execution metadata first, then images
        result = [
            {
                "session_id": session_id,
                "cell_index": data["cell_index"],
                "actual_notebook_index": data["actual_notebook_index"],
                "execution_count": data["execution_count"],
                "outputs": outputs,
            }
        ]
        result.extend(images)
        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to edit cell: {str(e)}")


@mcp.tool
async def shutdown_session(session_id: str) -> str:
    """
    Shutdown a kernel session gracefully.

    Note: using this tool terminates kernel state; it should typically only be used if the user
    has instructured you to do so.

    Args:  session_id: The session ID to shutdown
    """
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/shutdown",
            json={"session_id": session_id},
            headers=headers,
        )
        response.raise_for_status()

        # Clean up session images if image saving is enabled
        global _active_sessions
        return f"Session {session_id} shut down successfully"

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to shutdown session: {str(e)}")


@mcp.resource(
    uri="scribe://server/status",
    name="ScribeNotebookServerStatus",  # A human-readable name. If not provided, defaults to function name
    description="Get the current Scribe server status and connection information.",
)
async def server_status() -> str:
    status = get_server_status()

    # Format as a readable status report
    lines = [
        "# Scribe Server Status",
        "",
        f"**Status:** {status['status']}",
        f"**URL:** {status['url'] or 'Not available'}",
        f"**Port:** {status['port'] or 'Not available'}",
        f"**VSCode URL:** {status['vscode_url'] or 'Not available'}",
        f"**Health:** {status['health']}",
        f"**Auth Token:** {'Yes (auto-generated)' if get_token() else 'No'}",
        f"**External Server:** {'Yes' if status['is_external'] else 'No'}",
        f"**Will Shutdown on Exit:** {'Yes' if status['will_shutdown_on_exit'] else 'No'}",
    ]

    if not status["is_external"]:
        lines.extend(["", "*This server is automatically managed by the MCP server.*"])
    else:
        lines.extend(
            ["", "*This server was found via SCRIBE_PORT environment variable.*"]
        )

    return "\n".join(lines)


# Main entry point for STDIO transport
if __name__ == "__main__":
    mcp.run(transport="stdio")
