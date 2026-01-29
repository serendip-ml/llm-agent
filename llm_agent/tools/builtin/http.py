"""HTTP fetch tool."""

from __future__ import annotations

from typing import Any
from urllib.parse import ParseResult, urlparse

import httpx

from llm_agent.tools.base import BaseTool, ToolResult


class HTTPFetchTool(BaseTool):
    """Tool for fetching content from URLs.

    Allows the agent to retrieve content from HTTP/HTTPS URLs. Useful for
    fetching API responses, web pages, or other remote content.

    Security Note:
        This tool makes HTTP requests to external URLs. Consider restricting
        access using allowed_domains or blocked_domains in production.

    Example:
        tool = HTTPFetchTool(allowed_domains=["api.github.com"])
        result = tool.execute(url="https://api.github.com/repos/owner/repo")
        print(result.output)
    """

    name = "http_fetch"
    description = (
        "Fetch content from a URL via HTTP GET. "
        "Use for: retrieving API responses, fetching web pages, downloading text content. "
        "Returns the response body as text. "
        "Supports custom headers for authentication or content negotiation."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (must be http:// or https://)",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers to include in the request",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["url"],
    }

    def __init__(
        self,
        timeout: float = 30.0,
        max_response_size: int = 1000000,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize HTTP fetch tool.

        Args:
            timeout: Request timeout in seconds. Defaults to 30.
            max_response_size: Maximum response size in bytes. Defaults to 1MB.
            allowed_domains: If set, only these domains can be accessed.
                            Takes precedence over blocked_domains.
            blocked_domains: If set, these domains are blocked.
                            Ignored if allowed_domains is set.
            default_headers: Headers to include in all requests.
        """
        self._timeout = timeout
        self._max_response_size = max_response_size
        self._allowed_domains = set(allowed_domains) if allowed_domains else None
        self._blocked_domains = set(blocked_domains) if blocked_domains else None
        self._default_headers = default_headers or {}

    def execute(self, **kwargs: Any) -> ToolResult:
        """Fetch content from a URL.

        Args:
            **kwargs: Must contain 'url'. Optional: 'headers'.

        Returns:
            ToolResult with response body or error.
        """
        # Validate URL argument
        url = kwargs.get("url")
        if not isinstance(url, str) or not url:
            return ToolResult(success=False, output="", error="Missing or invalid 'url' argument")

        # Parse and validate URL
        parsed = self._parse_url(url)
        if isinstance(parsed, ToolResult):
            return parsed

        # Check domain restrictions
        if error := self._check_domain(parsed.netloc):
            return error

        # Validate headers
        headers = self._build_headers(kwargs.get("headers"))
        if isinstance(headers, ToolResult):
            return headers

        # Make the request
        return self._fetch(url, headers)

    def _parse_url(self, url: str) -> ParseResult | ToolResult:
        """Parse and validate URL. Returns ParseResult or error."""
        try:
            parsed = urlparse(url)
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Invalid URL: {e}")

        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid URL scheme '{parsed.scheme}'. Must be http or https.",
            )

        if not parsed.netloc:
            return ToolResult(success=False, output="", error="Invalid URL: missing domain")

        return parsed

    def _check_domain(self, domain: str) -> ToolResult | None:
        """Check if domain is allowed. Returns error if blocked."""
        # Normalize domain (remove port if present)
        domain_only = domain.split(":")[0].lower()

        if self._allowed_domains is not None:
            if not self._matches_domain(domain_only, self._allowed_domains):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Domain '{domain_only}' is not in allowed list",
                )
            return None

        if self._blocked_domains is not None and self._matches_domain(
            domain_only, self._blocked_domains
        ):
            return ToolResult(
                success=False,
                output="",
                error=f"Domain '{domain_only}' is blocked",
            )

        return None

    def _matches_domain(self, domain: str, domain_set: set[str]) -> bool:
        """Check if domain matches any in the set (including subdomains)."""
        for allowed in domain_set:
            allowed_lower = allowed.lower()
            if domain == allowed_lower:
                return True
            # Check subdomain match (e.g., "api.github.com" matches "github.com")
            if domain.endswith("." + allowed_lower):
                return True
        return False

    def _build_headers(self, custom_headers: Any) -> dict[str, str] | ToolResult:
        """Build request headers. Returns headers dict or error."""
        headers = dict(self._default_headers)

        if custom_headers is None:
            return headers

        if not isinstance(custom_headers, dict):
            return ToolResult(
                success=False,
                output="",
                error="Invalid 'headers' argument: must be an object",
            )

        for key, value in custom_headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return ToolResult(
                    success=False,
                    output="",
                    error="Invalid header: keys and values must be strings",
                )
            headers[key] = value

        return headers

    def _fetch(self, url: str, headers: dict[str, str]) -> ToolResult:
        """Perform the HTTP request."""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, headers=headers)
                return self._build_response(response)
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error=f"Request timed out after {self._timeout} seconds",
            )
        except httpx.ConnectError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Connection failed: {e}",
            )
        except httpx.TooManyRedirects:
            return ToolResult(
                success=False,
                output="",
                error="Too many redirects",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Request failed: {e}",
            )

    def _build_response(self, response: httpx.Response) -> ToolResult:
        """Build ToolResult from HTTP response."""
        if error := self._check_content_length(response):
            return error

        content_or_error = self._read_response_content(response)
        if isinstance(content_or_error, ToolResult):
            return content_or_error
        content, truncated = content_or_error

        return self._format_response(response, content, truncated)

    def _check_content_length(self, response: httpx.Response) -> ToolResult | None:
        """Check if Content-Length exceeds limit. Returns error or None."""
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self._max_response_size:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Response too large: {content_length} bytes (max {self._max_response_size})",
                    )
            except ValueError:
                pass  # Invalid Content-Length header, proceed to read and truncate if needed
        return None

    def _read_response_content(self, response: httpx.Response) -> tuple[str, bool] | ToolResult:
        """Read response body, truncating if needed. Returns (content, truncated) or error."""
        try:
            content = response.text
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Failed to read response: {e}")

        if len(content) > self._max_response_size:
            return content[: self._max_response_size], True
        return content, False

    def _format_response(
        self, response: httpx.Response, content: str, truncated: bool
    ) -> ToolResult:
        """Format the final response."""
        if not response.is_success:
            output = f"HTTP {response.status_code}\n\n{content}"
            if truncated:
                output += "\n\n(response truncated)"
            return ToolResult(
                success=False,
                output=output,
                error=f"HTTP {response.status_code}: {response.reason_phrase}",
            )

        output = content
        if truncated:
            output += f"\n\n(truncated, max {self._max_response_size} chars)"
        return ToolResult(success=True, output=output)
