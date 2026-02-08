"""
Tornado HTTP handlers for the Scribe Jupyter Server API.

This module contains all the HTTP request handlers that implement the Scribe API endpoints.
These handlers receive HTTP requests, call the appropriate methods on the ScribeServerApp,
and return JSON responses.

Each handler:
1. Validates the incoming request and extracts parameters
2. Calls the corresponding method on ScribeServerApp
3. Formats and returns the response as JSON
4. Handles errors gracefully with appropriate HTTP status codes
"""

import uuid
import json
from jupyter_server.base.handlers import APIHandler
from tornado.web import authenticated


# Tornado handlers for Scribe API
class ScribeAPIHandler(APIHandler):
    """Base handler for Scribe API endpoints."""

    @property
    def scribe_app(self):
        """Get the ScribeServerApp instance."""
        return self.settings["serverapp"]


class StartSessionHandler(ScribeAPIHandler):
    """Handler for starting sessions."""

    @authenticated
    async def post(self):
        request_id = str(uuid.uuid4())[:8]
        try:
            data = self.get_json_body() or {}

            result = await self.scribe_app.start_session(
                experiment_name=data.get("experiment_name"),
                existing_notebook_path=data.get("notebook_path"),
                fork_prev_notebook=data.get(
                    "fork_prev_notebook", True
                ),  # Default to True for backward compatibility
                kernel_name=data.get("kernel_name", "python3"),
            )

            # Add server URL and token to response
            result["server_url"] = f"http://localhost:{self.scribe_app.port}"
            result["token"] = self.scribe_app.token

            # Add VSCode URL
            result["vscode_url"] = (
                f"http://localhost:{self.scribe_app.port}/?token={self.scribe_app.token}"
            )
            result["kernel_display_name"] = result.pop("kernel_display_name", "")
            result["status"] = "started"

            self.finish(json.dumps(result))
        except Exception as e:
            self.set_status(500)
            self.finish(json.dumps({"error": str(e), "request_id": request_id}))


class ExecuteCodeHandler(ScribeAPIHandler):
    """Handler for executing code."""

    @authenticated
    async def post(self):
        request_id = str(uuid.uuid4())[:8]
        session_id = None
        try:
            data = self.get_json_body()
            if not data:
                raise ValueError("No JSON body provided")

            session_id = data.get("session_id")

            result = await self.scribe_app.execute_code_in_kernel(
                data["session_id"], data["code"]
            )

            # Add session_id to response
            result["session_id"] = data["session_id"]


            self.finish(json.dumps(result))
        except Exception as e:

            self.set_status(500)
            self.finish(json.dumps({"error": str(e), "request_id": request_id}))


class ShutdownSessionHandler(ScribeAPIHandler):
    """Handler for shutting down sessions."""

    @authenticated
    async def post(self):
        request_id = str(uuid.uuid4())[:8]
        session_id = None
        try:
            data = self.get_json_body()
            if not data:
                raise ValueError("No JSON body provided")

            session_id = data.get("session_id")

            await self.scribe_app.shutdown_session(data["session_id"])

            self.finish(json.dumps({"status": "shutdown"}))
        except Exception as e:
            self.set_status(500)
            self.finish(json.dumps({"error": str(e), "request_id": request_id}))


class AddMarkdownHandler(ScribeAPIHandler):
    """Handler for adding markdown cells."""

    @authenticated
    async def post(self):
        request_id = str(uuid.uuid4())[:8]
        session_id = None
        try:
            data = self.get_json_body()
            if not data:
                raise ValueError("No JSON body provided")

            session_id = data.get("session_id")

            cell_number = await self.scribe_app.add_markdown_cell(
                data["session_id"], data["content"]
            )


            self.finish(
                json.dumps(
                    {"session_id": data["session_id"], "cell_number": cell_number}
                )
            )

        except Exception as e:
            self.set_status(500)
            self.finish(json.dumps({"error": str(e), "request_id": request_id}))


class EditCellHandler(ScribeAPIHandler):
    """Handler for editing existing cells."""

    @authenticated
    async def post(self):
        request_id = str(uuid.uuid4())[:8]
        session_id = None
        try:
            data = self.get_json_body()
            if not data:
                raise ValueError("No JSON body provided")

            session_id = data.get("session_id")
            cell_index = data.get("cell_index", -1)  # Default to last cell
            new_code = data.get("code", "")

            # Edit the cell and get execution result
            result = await self.scribe_app.edit_and_execute_cell(
                session_id, cell_index, new_code
            )

            # Add session_id to response
            result["session_id"] = session_id


            self.finish(json.dumps(result))

        except Exception as e:
            self.set_status(500)
            self.finish(json.dumps({"error": str(e), "request_id": request_id}))


class HealthCheckHandler(ScribeAPIHandler):
    """Handler for health checks."""

    async def get(self):
        self.finish(
            json.dumps(
                {
                    "status": "healthy",
                    "server": "Scribe with Jupyter Server",
                    "jupyter_url": f"http://localhost:{self.scribe_app.port}/?token={self.scribe_app.token}",
                    "active_sessions": len(self.scribe_app.sessions),
                    "notebooks_dir": str(self.scribe_app.notebooks_path.absolute()),
                }
            )
        )


class TreeHandler(APIHandler):
    """Handler for /tree endpoint - VSCode compatibility."""

    async def get(self):
        # VSCode expects this endpoint to exist for server detection
        # Return a simple HTML response indicating the server is running
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Jupyter Server</title></head>
        <body>
            <h1>Scribe Jupyter Server</h1>
            <p>Server is running. Use the API endpoints for interaction.</p>
        </body>
        </html>
        """
        self.set_header("Content-Type", "text/html")
        self.finish(html)
