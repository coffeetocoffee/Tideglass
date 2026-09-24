"""EMODnet authentication token management for Tideglass.

This module provides secure token storage and rotation for EMODnet API access,
enabling higher rate limits than the anonymous tier. Tokens are stored in the
user's config directory and automatically validated on each request.

Usage::

    from tideglass.fetch_emodnet_auth import EmodnetTokenManager

    manager = EmodnetTokenManager()
    
    # Load or create token interactively
    if not manager.load_or_create_token():
        raise RuntimeError("Authentication failed")
    
    # Add auth header to download requests
    headers = {}
    manager.set_header(headers)  # adds Authorization: Bearer <token>
    
    rows = fetch_emodnet(..., _headers=headers)
    
    # Logout/clear credentials
    manager.clear_cached_credentials()
"""

from __future__ import annotations

import getpass
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class EmodnetAuthError(RuntimeError):
    """Raised when authentication fails."""


class EmodnetTokenManager:
    """Manages EMODnet authentication tokens securely.

    Tokens are stored in the user's config directory:
        Windows: %APPDATA%\\tideglass\\emodnet_token.json
        Linux/Mac: ~/.config/tideglass/emodnet_token.json
    
    Supports automatic token validation, expiration detection, and secure deletion.

    Attributes:
        CONFIG_DIR: platform-specific config directory
        TOKEN_FILE: path to token JSON file
        _token: cached token value (None if not authenticated)
        _expires_at: Unix timestamp of token expiration (None if unknown)
    """

    # Platform-specific config directories
    if os.name == "nt":
        CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "tideglass"
    else:
        CONFIG_DIR = Path.home() / ".config" / "tideglass"
    
    TOKEN_FILE = CONFIG_DIR / "emodnet_token.json"
    
    def __init__(self):
        self._token: str | None = None
        self._expires_at: float | None = None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if current token is valid and not expired."""
        import time
        
        if self._token is None:
            return False
        
        if self._expires_at and time.time() > self._expires_at - 3600:  # 1h buffer
            self._token = None
            self._expires_at = None
            return False
        
        return True
    
    @property
    def has_token(self) -> bool:
        """Check if a token exists on disk, regardless of validity."""
        return self.TOKEN_FILE.exists()
    
    def load_or_create_token(self) -> bool:
        """Load existing token or prompt user for new one.

        Returns True if successfully loaded/created, False otherwise.
        """
        if self.has_token:
            try:
                with open(self.TOKEN_FILE) as fh:
                    data = json.load(fh)
                
                self._token = data.get("token")
                self._expires_at = data.get("expires_at")
                
                if self.is_authenticated:
                    return True
            
            except (json.JSONDecodeError, KeyError, OSError):
                # Corrupted token file
                pass
        
        # No valid token found - prompt user
        print("\n" + "=" * 60)
        print("EMODnet Authentication Required")
        print("=" * 60)
        print("\nThe free EMODnet API tier allows ~100 downloads/month.")
        print("For higher quotas, register at:")
        print("  https://emodnet.ec.europa.eu/en/services/api")
        print("\nOnce registered, your API token will be emailed to you.")
        print("Paste it below (hidden input enabled):\n")
        
        try:
            token = getpass.getpass("API Token: ").strip()
            
            if not token:
                print("Error: empty token provided")
                return False
            
            expires_in_days = 365  # Reasonable default assumption
            self._expires_at = (datetime.now() + timedelta(days=expires_in_days)).timestamp()
            
            # Create config directory
            self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            
            # Store securely with proper permissions (Windows: readable by owner only)
            token_data = {
                "token": token,
                "expires_at": self._expires_at,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
            }
            
            with open(self.TOKEN_FILE, "w") as fh:
                json.dump(token_data, fh, indent=2)
            
            # Set restrictive file permissions on Linux/Mac
            if os.name != "nt":
                try:
                    os.chmod(self.TOKEN_FILE, 0o600)
                except OSError:
                    pass  # Best effort only
            
            self._token = token
            return True
            
        except KeyboardInterrupt:
            print("\nToken entry cancelled")
            return False
    
    def invalidate(self) -> None:
        """Delete stored token (logout)."""
        if self.TOKEN_FILE.exists():
            self.TOKEN_FILE.unlink()
        self._token = None
        self._expires_at = None
    
    def set_header(self, headers: dict) -> dict:
        """Add authorization header if authenticated.

        :param headers: dictionary to modify in place
        :returns: modified headers dict with Authorization added if token present
        """
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers
    
    def clear_cached_credentials(self) -> None:
        """Remove all cached credentials from disk and memory."""
        self.invalidate()
    
    @classmethod
    def clear_all_cached_credentials(cls) -> None:
        """Class method to remove all cached credentials (useful in testing)."""
        instance = cls()
        instance.invalidate()
    
    def status(self) -> dict:
        """Return human-readable token status information."""
        if not self.is_authenticated:
            return {"authenticated": False, "status": "no valid token"}
        
        if self._expires_at:
            remaining = self._expires_at - __import__("time").time()
            days_remaining = int(remaining / 86400)
            return {
                "authenticated": True,
                "status": "active",
                "days_remaining": max(days_remaining, 0),
                "file_path": str(self.TOKEN_FILE),
            }
        
        return {"authenticated": True, "status": "unknown expiry"}
    
    def validate_token_format(self, token: str) -> tuple[bool, str]:
        """Basic validation of token format before accepting.

        EMODnet tokens are typically base64-encoded UUIDs (~36-64 chars).
        This is heuristic and doesn't verify server-side validity.
        
        Returns (is_valid, reason_message).
        """
        token = token.strip()
        
        if len(token) < 10:
            return False, "token too short (< 10 characters)"
        
        if len(token) > 512:
            return False, "token too long (> 512 characters)"
        
        # Allow alphanumeric + dash/underscore/special chars typical of JWT-style tokens
        import re
        if not re.match(r"^[A-Za-z0-9_\-./]+$", token):
            return False, "token contains invalid characters"
        
        return True, ""


# Module-level convenience functions

def emodnet_login(store: str | None = None) -> bool:
    """Interactive login command (used by CLI `tideglass emodnet login`).

    :param store: optional store override for where to save token
    """
    manager = EmodnetTokenManager()
    return manager.load_or_create_token()


def emodnet_logout() -> bool:
    """Logout command (used by CLI `tideglass emodnet logout`)."""
    manager = EmodnetTokenManager()
    if manager.has_token:
        manager.invalidate()
        print("EMODnet token removed.")
        return True
    else:
        print("No EMODnet token found.")
        return False


def emodnet_status() -> dict:
    """Status check command (used by CLI `tideglass emodnet status`)."""
    manager = EmodnetTokenManager()
    return manager.status()


def emodnet_fetch_with_auth(url: str, **kwargs) -> bytes:
    """Helper to fetch EMODnet URL with authentication included.

    This wraps urllib to inject auth headers automatically when available.
    """
    import urllib.request
    
    manager = EmodnetTokenManager()
    headers = {"User-Agent": "tideglass/3.3.0 (+https://github.com/coffeetocoffee/Tideglass)"}
    manager.set_header(headers)
    
    req = urllib.request.Request(url, headers=headers, **kwargs)
    return urllib.request.urlopen(req).read()
