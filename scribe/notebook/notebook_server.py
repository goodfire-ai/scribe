"""
Runs a Jupyter Server for the Scribe MCP server to connect to.
"""

import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys
import os

import nbformat
from jupyter_server.serverapp import ServerApp
from traitlets import Unicode, Int
from scribe.notebook._notebook_server_utils import (
    clean_notebook_for_save,
    get_notebook_metadata_for_kernel,
)
from dataclasses import dataclass

from . import notebook_sever_handlers as _handlers



# Request/Response models as simple dicts for Tornado handlers
@dataclass
class ScribeNotebookSession:
    """Container for all session-related data."""

    session_id: str
    kernel_id: str
    jupyter_session_id: str
    notebook_path: Path
    display_name: str
    kernel_name: str = "python3"
    execution_count: int = 0
    last_activity: Optional[datetime] = None


class ScribeServerApp(ServerApp):
    """Jupyter Server app with Scribe customizations."""

    notebooks_dir = Unicode(
        "notebooks",
        config=True,
        help="Directory for saving notebooks. Supports ~ expansion and environment variables.",
    )

    auto_shutdown_minutes = Int(
        60, config=True, help="Minutes before auto-shutdown of idle kernels"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Single source of truth for all session data
        # Maps session ID to session instance
        self.sessions: Dict[str, ScribeNotebookSession] = {}

        # Track last activity time
        self.last_activity_time = datetime.now()

        # Set up auto-shutdown timer if enabled
        if self.auto_shutdown_minutes > 0:
            from tornado.ioloop import PeriodicCallback

            self.shutdown_check_callback = PeriodicCallback(
                self.check_auto_shutdown,
                60000,  # Check every minute
            )

        # Note: notebooks_path will be set up in initialize() after config is parsed
        self.notebooks_path = None

    def initialize(self, argv=None):
        """Initialize the server after parsing configuration."""
        # Call parent initialization first to parse config
        super().initialize(argv)

        # Now set up notebooks directory with the parsed configuration
        self.notebooks_path = self._setup_notebooks_directory()


    def _setup_notebooks_directory(self) -> Path:
        """Set up and validate the notebooks directory with enhanced path handling."""
        try:
            # Expand user home directory (~) and environment variables
            expanded_path = os.path.expanduser(os.path.expandvars(self.notebooks_dir))

            # Convert to Path object
            notebooks_path = Path(expanded_path)

            # Make it absolute if it's relative
            if not notebooks_path.is_absolute():
                notebooks_path = Path.cwd() / notebooks_path

            # Resolve any relative components (like .. or .)
            notebooks_path = notebooks_path.resolve()

            # Check if path exists and is a file (not allowed)
            if notebooks_path.exists() and notebooks_path.is_file():
                raise ValueError(
                    f"Notebooks directory path '{notebooks_path}' exists but is a file, not a directory"
                )

            # Create directory if it doesn't exist
            notebooks_path.mkdir(parents=True, exist_ok=True)

            # Verify we can write to the directory
            test_file = notebooks_path / ".scribe_write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except PermissionError:
                raise ValueError(
                    f"No write permission for notebooks directory: {notebooks_path}"
                )

            print(f"📁 Notebooks directory: {notebooks_path}")
            return notebooks_path

        except Exception as e:
            error_msg = (
                f"Failed to set up notebooks directory '{self.notebooks_dir}': {str(e)}"
            )
            print(f"❌ {error_msg}")
            raise ValueError(error_msg) from e

    async def _get_kernel_language_info(self, kernel_id: str) -> dict:
        """Query a running kernel for its language_info via kernel_info_request.

        Returns the full language_info dict (name, version, mimetype, etc.)
        or a minimal fallback if the request times out.
        """
        kernel = self.kernel_manager.get_kernel(kernel_id)
        client = kernel.client()
        client.start_channels()
        try:
            msg = await client._async_kernel_info()
            return msg.get("content", {}).get("language_info", {})
        except Exception as e:
            print(f"Warning: failed to query language_info from kernel {kernel_id}: {e}")
            return {}
        finally:
            client.stop_channels()

    def init_webapp(self):
        """Add our custom handlers to the web app."""
        super().init_webapp()

        # Add Scribe API handlers
        host_pattern = ".*$"
        base_url = self.base_url

        handlers = [
            (f"{base_url}api/scribe/start", _handlers.StartSessionHandler),
            (f"{base_url}api/scribe/exec", _handlers.ExecuteCodeHandler),
            (f"{base_url}api/scribe/shutdown", _handlers.ShutdownSessionHandler),
            (f"{base_url}api/scribe/markdown", _handlers.AddMarkdownHandler),
            (f"{base_url}api/scribe/edit", _handlers.EditCellHandler),
            (f"{base_url}api/scribe/health", _handlers.HealthCheckHandler),
            # VSCode compatibility - it expects /tree endpoint
            (f"{base_url}tree", _handlers.TreeHandler),
        ]

        self.web_app.add_handlers(host_pattern, handlers)

        # Start the auto-shutdown timer after webapp is initialized
        if hasattr(self, "shutdown_check_callback"):
            self.shutdown_check_callback.start()

    async def start_session(
        self, experiment_name=None, existing_notebook_path=None, fork_prev_notebook=True, kernel_name="python3",
    ):
        """
        Start a new scribe jupyter session -- one-to-one with a notebook and a kernel.

        Args:
            experiment_name: Name for the experiment/notebook
            existing_notebook_path: Path to an existing notebook to use, either as a copy/fork, or to directly resume from
            fork_prev_notebook: If True (default) and existing_notebook_path is provided, creates a new
                          notebook copying the existing one. If False, uses the existing
                          notebook in-place. Both options re-run all cells.
                          NOTE: re-running cells without forking will update existing file, and may overwrite outputs
                          e.g. with different random outputs if seeds have not been set, or with errors if incorrect env
                          is being used.
            kernel_name: Jupyter kernel spec name (e.g. "python3", "ir", "julia-1.12"). Must be installed and registered with Jupyter.
        """
        # Ensure notebooks directory is set up
        if self.notebooks_path is None:
            self.notebooks_path = self._setup_notebooks_directory()

        # Update activity timestamp
        self.update_activity()

        # Generate new session ID
        session_id = str(uuid.uuid4())

        """STEP 1: CREATE NOTEBOOK FILE"""
        # Determine the notebook path and whether we're creating a new file
        if existing_notebook_path:
            source_path = Path(existing_notebook_path)
            if not source_path.exists():
                raise ValueError(f"Notebook not found: {existing_notebook_path}")

            if fork_prev_notebook:
                # Fork: Create a new notebook file
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
                original_name = source_path.stem
                base_name = f"{timestamp}_{experiment_name or original_name}_continued"

                # Use notebook directory
                target_dir = self.notebooks_path
                target_dir.mkdir(parents=True, exist_ok=True)

                # Find available filename. Add _{idx} until we've created a unique name.
                nb_path = target_dir / f"{base_name}.ipynb"
                counter = 1
                while nb_path.exists():
                    nb_path = target_dir / f"{base_name}_{counter}.ipynb"
                    counter += 1

                # Copy the notebook
                import shutil

                shutil.copy2(source_path, nb_path)

                kernel_display_name = f"Scribe: {base_name}"
            else:
                # Resume: Use existing notebook
                nb_path = source_path
                kernel_display_name = f"Resumed: {nb_path.name}"
        else:
            # Create new empty notebook
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            base_name = f"{timestamp}_{experiment_name or 'Notebook'}"

            # Use notebook directory
            target_dir = self.notebooks_path
            target_dir.mkdir(parents=True, exist_ok=True)

            # Find available filename
            nb_path = target_dir / f"{base_name}.ipynb"
            counter = 1
            while nb_path.exists():
                nb_path = target_dir / f"{base_name}_{counter}.ipynb"
                counter += 1

            # Create empty notebook
            nb = nbformat.v4.new_notebook()
            kernel_meta = get_notebook_metadata_for_kernel(
                self.kernel_spec_manager, kernel_name
            )
            nb.metadata.update(kernel_meta)

            # Save notebook
            with open(nb_path, "w") as f:
                nbformat.write(clean_notebook_for_save(nb), f)

            kernel_display_name = f"Scribe: {base_name}"

        """STEP 2: Create a Jupyter session """
        # Note: v0 os scribe created just a kernel, but using a full jupyter session lets
        # us connect to kernels via VSCode and offers additional features for free.
        try:
            relative_path = nb_path.relative_to(Path.cwd())
        except ValueError:
            # If notebook is outside cwd, use absolute path
            relative_path = nb_path

        # Create a kernel first to ensure it uses our current environment
        kernel_id = await self.kernel_manager.start_kernel(kernel_name=kernel_name)

        # Now create a session and associate it with our kernel
        sm = self.web_app.settings["session_manager"]
        session = await sm.create_session(
            path=str(relative_path),
            type="notebook",
            name=nb_path.name,  # VSCode requires this field
            kernel_id=kernel_id,  # Use our kernel with correct environment
        )
        jupyter_session_id = session["id"]

        # Store session data in single container
        scribe_session = ScribeNotebookSession(
            session_id=session_id,  # internal scribe session ID
            kernel_id=kernel_id,
            jupyter_session_id=jupyter_session_id,
            notebook_path=nb_path,
            display_name=kernel_display_name,
            kernel_name=kernel_name,
            last_activity=datetime.now(),
        )
        self.sessions[session_id] = scribe_session

        # Update notebook with kernel info after beginning
        language_info = await self._get_kernel_language_info(kernel_id)
        if language_info:
            with open(nb_path, "r") as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            nb.metadata["language_info"] = language_info
            with open(nb_path, "w") as f:
                nbformat.write(clean_notebook_for_save(nb), f)

        """STEP 3: If we have an existing notebook, execute all code cells to restore state """
        restoration_results = []
        if existing_notebook_path:
            # Read the notebook
            with open(nb_path, "r") as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

            if nb.cells:
                # Use the same approach for both fork and resume: update cells in place
                print(
                    f"{'Restoring' if fork_prev_notebook else 'Resuming'} notebook with {len(nb.cells)} cells..."
                )

                for i, cell in enumerate(nb.cells):
                    if cell.cell_type == "code" and cell.source.strip():
                        try:
                            # Clear existing outputs in the cell
                            cell.outputs = []

                            # Execute the cell
                            print(f"Executing cell {i + 1} (code cell)...")

                            # Get next execution count
                            scribe_session.execution_count += 1
                            execution_count = scribe_session.execution_count
                            cell.execution_count = execution_count

                            # Execute and collect outputs
                            outputs = []
                            async for output in self._execute_and_stream(
                                session_id, cell.source
                            ):
                                outputs.append(output)

                                # Convert output to nbformat and add to cell
                                if output["output_type"] == "stream":
                                    cell_output = nbformat.v4.new_output(
                                        output_type="stream",
                                        name=output["name"],
                                        text=output["text"],
                                    )
                                elif output["output_type"] == "execute_result":
                                    cell_output = nbformat.v4.new_output(
                                        output_type="execute_result",
                                        data=output["data"],
                                        metadata=output.get("metadata", {}),
                                        execution_count=output.get("execution_count"),
                                    )
                                elif output["output_type"] == "display_data":
                                    cell_output = nbformat.v4.new_output(
                                        output_type="display_data",
                                        data=output["data"],
                                        metadata=output.get("metadata", {}),
                                    )
                                elif output["output_type"] == "error":
                                    cell_output = nbformat.v4.new_output(
                                        output_type="error",
                                        ename=output["ename"],
                                        evalue=output["evalue"],
                                        traceback=output["traceback"],
                                    )
                                else:
                                    continue

                                cell.outputs.append(cell_output)

                            # Check if there were any errors in the outputs
                            errors = [
                                o for o in outputs if o.get("output_type") == "error"
                            ]
                            if errors:
                                error_msg = f"Cell {i + 1} executed with errors: {errors[0]['ename']}: {errors[0]['evalue']}"
                                print(f"ERROR: {error_msg}")
                                restoration_results.append(
                                    {
                                        "cell": i + 1,
                                        "status": "error",
                                        "error": error_msg,
                                        "traceback": errors[0].get("traceback", []),
                                    }
                                )
                            else:
                                print(f"✓ Cell {i + 1} executed successfully")
                                restoration_results.append(
                                    {"cell": i + 1, "status": "success"}
                                )

                        except Exception as e:
                            error_msg = f"Failed to execute cell {i + 1}: {str(e)}"
                            print(f"ERROR: {error_msg}")
                            restoration_results.append(
                                {"cell": i + 1, "status": "error", "error": error_msg}
                            )
                            # Continue with other cells even if one fails

                # Save the notebook with updated outputs
                with open(nb_path, "w") as f:
                    nbformat.write(clean_notebook_for_save(nb), f)

        result = {
            "session_id": session_id,
            "kernel_id": kernel_id,
            "notebook_path": str(nb_path),
            "kernel_display_name": kernel_display_name,
            "kernel_name": kernel_name,
        }

        # Include restoration results in output if we restored from an existing notebook
        # e.g. so agent can see whether errors occurred.
        if restoration_results:
            result["restoration_results"] = restoration_results
            failed_cells = [r for r in restoration_results if r["status"] == "error"]
            operation = "restored" if fork_prev_notebook else "resumed"
            if failed_cells:
                result["restoration_summary"] = (
                    f"{operation.capitalize()} with {len(failed_cells)} errors out of {len(restoration_results)} cells"
                )
            else:
                result["restoration_summary"] = (
                    f"Successfully {operation} all {len(restoration_results)} cells"
                )

        return result

    async def add_markdown_cell(self, session_id: str, content: str):
        """Add a markdown cell to the notebook."""
        self.update_session_activity(session_id)
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Read notebook
        with open(session.notebook_path, "r") as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        # Add markdown cell
        cell = nbformat.v4.new_markdown_cell(source=content)
        nb.cells.append(cell)

        # Write back
        with open(session.notebook_path, "w") as f:
            nbformat.write(clean_notebook_for_save(nb), f)

        return len(nb.cells)

    async def _add_pending_cell(self, session_id: str, code: str) -> int:
        """
        Add a code cell with pending execution status.
        Used to immediately add a code cell to the notebook file before execution
        begins, giving users visual feedback that something is happening.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Read notebook
        with open(session.notebook_path, "r") as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        # Get next execution count
        session.execution_count += 1
        execution_count = session.execution_count

        # Create new cell with pending status
        cell = nbformat.v4.new_code_cell(
            source=code,
            outputs=[],
            execution_count=execution_count,
            metadata={"execution_status": "pending"},
        )

        # Append cell
        nb.cells.append(cell)
        cell_index = len(nb.cells) - 1

        # Write back immediately
        with open(session.notebook_path, "w") as f:
            nbformat.write(clean_notebook_for_save(nb), f)

        return cell_index

    async def _update_cell_output(
        self, session_id: str, cell_index: int, output: dict, status: str = "running"
    ):
        """Update a cell with a new output."""
        session = self.sessions.get(session_id)
        if not session:
            return

        # Read notebook
        with open(session.notebook_path, "r") as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        if cell_index >= len(nb.cells):
            return

        cell = nb.cells[cell_index]

        # Convert output dict to nbformat output
        if output["output_type"] == "stream":
            cell_output = nbformat.v4.new_output(
                output_type="stream", name=output["name"], text=output["text"]
            )
        elif output["output_type"] == "execute_result":
            cell_output = nbformat.v4.new_output(
                output_type="execute_result",
                data=output["data"],
                metadata=output.get("metadata", {}),
                execution_count=output.get("execution_count"),
            )
        elif output["output_type"] == "display_data":
            cell_output = nbformat.v4.new_output(
                output_type="display_data",
                data=output["data"],
                metadata=output.get("metadata", {}),
            )
        elif output["output_type"] == "error":
            cell_output = nbformat.v4.new_output(
                output_type="error",
                ename=output["ename"],
                evalue=output["evalue"],
                traceback=output["traceback"],
            )
        else:
            return

        # Append output
        cell.outputs.append(cell_output)

        # Update status
        cell.metadata["execution_status"] = status

        # Write back
        with open(session.notebook_path, "w") as f:
            nbformat.write(clean_notebook_for_save(nb), f)

    async def _execute_and_stream(self, session_id: str, code: str):
        """Execute code and yield outputs as they arrive."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        kernel_id = session.kernel_id

        # Create a client to execute code
        kernel = self.kernel_manager.get_kernel(kernel_id)
        client = kernel.client()
        client.start_channels()

        try:
            # Execute code
            msg_id = client.execute(code)
            execution_count = None

            # Stream outputs as they arrive
            while True:
                try:
                    # Use async get_iopub_msg
                    msg = await client._async_get_iopub_msg(
                        timeout=300
                    )  # 5 minute timeout

                    if msg["parent_header"].get("msg_id") != msg_id:
                        continue

                    msg_type = msg["msg_type"]
                    content = msg["content"]

                    if msg_type == "execute_input":
                        execution_count = content["execution_count"]
                        session.execution_count = execution_count
                    elif msg_type == "stream":
                        yield {
                            "output_type": "stream",
                            "name": content["name"],
                            "text": content["text"],
                        }
                    elif msg_type == "execute_result":
                        yield {
                            "output_type": "execute_result",
                            "data": content["data"],
                            "metadata": content.get("metadata", {}),
                            "execution_count": content["execution_count"],
                        }
                    elif msg_type == "display_data":
                        yield {
                            "output_type": "display_data",
                            "data": content["data"],
                            "metadata": content.get("metadata", {}),
                        }
                    elif msg_type == "error":
                        yield {
                            "output_type": "error",
                            "ename": content["ename"],
                            "evalue": content["evalue"],
                            "traceback": content["traceback"],
                        }
                    elif msg_type == "status" and content["execution_state"] == "idle":
                        break

                except Exception as e:
                    print(f"Error collecting output: {e}")
                    break

        finally:
            client.stop_channels()

    async def execute_code_in_kernel(
        self, session_id: str, code: str, skip_notebook_update: bool = False
    ):
        """Execute code in a kernel with immediate notebook updates.

        Args:
            session_id: The session ID
            code: The code to execute
            skip_notebook_update: If True, skip showing pending results as they're being computed
        """
        # Update activity timestamp
        self.update_activity()
        self.update_session_activity(session_id)

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Phase 1: Add cell to notebook immediately for quick visual feedback
        # (Unless we're restoring -- in that case we just add code after it finishes running for simplicity)
        if not skip_notebook_update:
            cell_index = await self._add_pending_cell(session_id, code)
        else:
            cell_index = None

        # Phase 2: Execute and stream outputs
        outputs = []
        execution_count = session.execution_count

        try:
            async for output in self._execute_and_stream(session_id, code):
                outputs.append(output)
                # Phase 3: Update notebook with each output as it arrives (only if we added a cell)
                if cell_index is not None:
                    await self._update_cell_output(
                        session_id, cell_index, output, status="running"
                    )

            # Mark as complete
            if cell_index is not None:
                await self._update_cell_status(session_id, cell_index, "complete")

        except Exception as e:
            # If error occurs, mark cell as error
            error_output = {
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [f"Error during execution: {str(e)}"],
            }
            outputs.append(error_output)
            if cell_index is not None:
                await self._update_cell_output(
                    session_id, cell_index, error_output, status="error"
                )

        return {
            "outputs": outputs,
            "execution_count": execution_count,
            "cell_index": cell_index,
        }

    async def _update_cell_status(self, session_id: str, cell_index: int, status: str):
        """Update just the status of a cell."""
        session = self.sessions.get(session_id)
        if not session:
            return

        # Read notebook
        with open(session.notebook_path, "r") as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        if cell_index < len(nb.cells):
            nb.cells[cell_index].metadata["execution_status"] = status

            # Write back
            with open(session.notebook_path, "w") as f:
                nbformat.write(clean_notebook_for_save(nb), f)

    async def edit_and_execute_cell(
        self, session_id: str, cell_index: int, new_code: str
    ):
        """Edit an existing cell and execute the new code."""
        # Update activity timestamp
        self.update_activity()
        self.update_session_activity(session_id)

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Read notebook
        with open(session.notebook_path, "r") as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        # Find code cells only
        code_cells = [
            (i, cell) for i, cell in enumerate(nb.cells) if cell.cell_type == "code"
        ]

        if not code_cells:
            raise ValueError("No code cells found in notebook")

        # Handle negative indexing for convenience
        if cell_index < 0:
            cell_index = len(code_cells) + cell_index

        if cell_index < 0 or cell_index >= len(code_cells):
            raise ValueError(
                f"Cell index {cell_index} out of range. Notebook has {len(code_cells)} code cells."
            )

        # Get the actual notebook index and cell
        actual_index, cell = code_cells[cell_index]

        # Update the cell source and clear outputs
        cell.source = new_code
        cell.outputs = []

        # Get next execution count
        session.execution_count += 1
        execution_count = session.execution_count
        cell.execution_count = execution_count
        cell.metadata["execution_status"] = "pending"

        # Save the updated cell immediately
        with open(session.notebook_path, "w") as f:
            nbformat.write(clean_notebook_for_save(nb), f)

        # Execute and stream outputs
        outputs = []
        try:
            async for output in self._execute_and_stream(session_id, new_code):
                outputs.append(output)
                # Update notebook with each output as it arrives
                await self._update_cell_output(
                    session_id, actual_index, output, status="running"
                )

            # Mark as complete
            await self._update_cell_status(session_id, actual_index, "complete")

        except Exception as e:
            # If error occurs, mark cell as error
            error_output = {
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [f"Error during execution: {str(e)}"],
            }
            outputs.append(error_output)
            await self._update_cell_output(
                session_id, actual_index, error_output, status="error"
            )

        return {
            "cell_index": cell_index,
            "actual_notebook_index": actual_index,
            "outputs": outputs,
            "execution_count": execution_count,
        }

    async def shutdown_session(self, session_id: str):
        """Shutdown a session and its kernel."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Delete the Jupyter session (this also shuts down the kernel)
        sm = self.web_app.settings["session_manager"]
        await sm.delete_session(session.jupyter_session_id)


        # Clean up our session tracking
        del self.sessions[session_id]

    def update_activity(self):
        """Update the last activity timestamp for entire ServerApp instance."""
        self.last_activity_time = datetime.now()

    def update_session_activity(self, session_id: str):
        """Update the last activity timestamp for a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now()

    async def check_auto_shutdown(self):
        """Check if server should auto-shutdown due to inactivity."""
        current_time = datetime.now()

        # First, clean up inactive sessions
        inactive_sessions = []
        for session_id, session in self.sessions.items():
            session_idle_minutes = (
                current_time - session.last_activity
            ).total_seconds() / 60
            if session_idle_minutes >= self.auto_shutdown_minutes:
                inactive_sessions.append(session_id)

        # Remove inactive sessions
        for session_id in inactive_sessions:
            print(
                f"🧹 Cleaning up inactive session {session_id} (idle for {self.auto_shutdown_minutes}+ minutes)"
            )
            try:
                await self.shutdown_session(session_id)
            except Exception as e:
                print(f"   Error shutting down session {session_id}: {e}")
                # Still remove from tracking even if shutdown fails
                if session_id in self.sessions:
                    del self.sessions[session_id]

        # Now check if we should shutdown the server
        if not self.sessions:  # No active sessions remaining
            idle_minutes = (current_time - self.last_activity_time).total_seconds() / 60

            if idle_minutes >= self.auto_shutdown_minutes:
                print(
                    f"\n⏰ Auto-shutdown: Server idle for {int(idle_minutes)} minutes"
                )
                print("   Shutting down...")

                # Stop the periodic callback
                if hasattr(self, "shutdown_check_callback"):
                    self.shutdown_check_callback.stop()

                # Gracefully stop the server
                self.stop()

    def cleanup(self):
        """Clean up resources when server is shutting down."""
        # Call parent cleanup
        super().cleanup()


if __name__ == "__main__":
    """Entry point for running the Scribe server as a script."""
    import sys

    # Create and configure the server app
    app = ScribeServerApp()

    # Parse command line arguments
    if len(sys.argv) > 1:
        app.initialize(sys.argv[1:])
    else:
        app.initialize()

    # Start the server
    app.start()
