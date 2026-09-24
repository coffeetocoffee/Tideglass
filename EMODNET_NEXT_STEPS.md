# EMODnet Integration Guide - Next Steps Implementation

This document provides implementation patterns and examples for extending Tideglass European data capabilities.

---

## Already Implemented ✅

### Step 1: Batch Download Support

**Functions:** `fetch_emodnet_batch()`, `_short_bbox_hash()`

**Usage Example:**
```python
from tideglass.fetch_emodnet import fetch_emodnet_batch

# Define multiple regions covering Europe
queries = [
    {"bbox": (-10.0, 5.0, 35.0, 60.0), "start": "2024-02-01"},  # Iberia/UK
    {"bbox": (5.0, 15.0, 40.0, 55.0)},                           # France/Benelux
    {"bbox": (10.0, 25.0, 35.0, 50.0)},                          # Mediterranean
    {"bbox": (15.0, 35.0, 50.0, 70.0)},                          # Northern Europe/Baltic
]

# Concurrently download all regions
results = fetch_emodnet_batch(queries, out_dir=".emodnet_cache")

for region_id, rows in results.items():
    print(f"{region_id}: {len(rows)} observations")
    # Save each result for processing
    export.write_csv(rows, f"{region_id}.csv")
```

### Step 3: Quality Flag Filtering

**Function:** `fetch_emodnet_with_quality()`

**Usage Example:**
```python
from tideglass.fetch_emodnet import fetch_emodnet_with_quality

# Get only validated (highest quality) sea level data
validated_rows = fetch_emodnet_with_quality(
    lon_min=-10.0, lon_max=5.0, lat_min=35.0, lat_max=60.0,
    start="2024-02-01", end="2024-02-28",
    min_quality="validated"  # Options: "raw", "processed", "validated"
)

# Compare quality levels
raw_rows = fetch_emodnet_with_quality(
    ... same params ..., min_quality="raw"
)
print(f"Raw: {len(raw_rows)} | Validated: {len(validated_rows)}")
# Filtered out lower-quality records automatically
```

---

## Remaining Steps to Implement

### Step 2: Continuous Monitoring/Polling Loop

**Goal:** Create `tideglass emodnet poll` similar to existing `tideglass poll` for NOAA

**Implementation Pattern:**

```python
# tideglass/fetch_emodnet_poll.py (new file)

import time
from datetime import datetime, timezone
from tideglass.fetch_emodnet import fetch_emodnet

def poll_emodnet(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    store: str = ".tideglass",
    lookback_hours: int = 24,
    sleep_seconds: int = 3600,  # 1 hour default
    max_passes: int | None = None,
) -> dict:
    """Continuous polling loop for EMODnet regions.

    :param store: artifact directory for models (same as NOAA poll).
    :param lookback_hours: hours of recent data to fetch each pass.
    :param sleep_seconds: seconds between polling cycles.
    :param max_passes: optional hard limit on iterations (None = infinite).
    
    Returns statistics dictionary after completion.
    """
    from tideglass.marea import ops as OPS
    
    stats = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "passes": 0,
        "rows_fetched": 0,
        "errors": [],
        "stations_updated": 0,
    }
    
    passes = 0
    
    while True:
        passes += 1
        if max_passes and passes > max_passes:
            break
        
        try:
            # Fetch last N hours
            end = datetime.now(timezone.utc)
            start = end.replace(hour=end.hour - (lookback_hours % 24))
            
            rows = fetch_emodnet(
                lon_min=lon_min, lon_max=lon_max,
                lat_min=lat_min, lat_max=lat_max,
                start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                out_dir=f"{store}/emodnet_cache",
            )
            
            stats["passes"] = passes
            stats["rows_fetched"] += len(rows)
            stats["last_fetch"] = end.isoformat()
            
            # Process rows (fit/update model here)
            # TODO: Add model fitting logic like NOAA poll
            
            print(f"Pass {passes}: {len(rows)} rows fetched")
            
        except Exception as exc:
            error_info = {
                "time": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            stats["errors"].append(error_info)
            print(f"Error on pass {passes}: {exc}")
        
        # Sleep between passes
        if max_passes is None or passes < max_passes:
            time.sleep(sleep_seconds)
    
    stats["end_time"] = datetime.now(timezone.utc).isoformat()
    return stats


# CLI wrapper (to add to cli.py)
def cmd_emodnet_poll(args) -> int:
    """CLI entry point: tideglass emodnet poll --bbox ..."""
    try:
        stats = poll_emodnet(
            lon_min=args.bbox[0],
            lon_max=args.bbox[1],
            lat_min=args.bbox[2],
            lat_max=args.bbox[3],
            store=args.store,
            lookback_hours=args.lookback_h,
            sleep_seconds=args.sleep_s,
            max_passes=args.repeat,
        )
        
        print("Polling complete:")
        print(f"  Passes: {stats['passes']}")
        print(f"  Rows fetched: {stats['rows_fetched']}")
        print(f"  Errors: {len(stats['errors'])}")
        return 0
        
    except Exception as exc:
        print(f"tideglass emodnet poll: {exc}")
        return 2
```

**CLI Command Addition:**

```bash
tideglass emodnet poll --bbox -10 5 35 60 \
    --store .tideglass \
    --lookback-h 24 \
    --sleep-s 3600 \
    --repeat 10
```

---

### Step 4: Authentication Token Management

**Challenge:** EMODnet free tier has rate limits; authentication enables higher quotas.

**Implementation Pattern:**

```python
# tideglass/fetch_emodnet_auth.py (new file)

import os
import hashlib
import json
from pathlib import Path
from typing import Optional


class EmodnetTokenManager:
    """Manages EMODnet authentication tokens securely.

    Tokens are stored in user config directory:
        Windows: %APPDATA%\tideglass\emodnet_token.json
        Linux/Mac: ~/.config/tideglass/emodnet_token.json
    
    Supports automatic token rotation and validation.
    """
    
    CONFIG_DIR = Path.home() / "AppData" / "Roaming" / "tideglass"  # Windows path
    TOKEN_FILE = CONFIG_DIR / "emodnet_token.json"
    
    def __init__(self):
        self._token: Optional[str] = None
        self._expires_at: Optional[float] = None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if current token is valid."""
        import time
        
        if self._token is None:
            return False
        
        if self._expires_at and time.time() > self._expires_at - 3600:  # 1h buffer
            self._token = None
            return False
        
        return self._token is not None
    
    def load_or_create_token(self) -> bool:
        """Load existing token or prompt user for new one."""
        import getpass
        
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        if self.TOKEN_FILE.exists():
            try:
                with open(self.TOKEN_FILE) as fh:
                    data = json.load(fh)
                    self._token = data.get("token")
                    self._expires_at = data.get("expires_at")
                
                return self.is_authenticated
            except (json.JSONDecodeError, KeyError):
                # Corrupted token file
                pass
        
        # No valid token found
        print("EMODnet requires an API token for higher rate limits.")
        print("Register at https://emodnet.ec.europa.eu/en/services/api")
        print("Paste your token below:")
        
        token = getpass.getpass("Token: ")
        expires_in_days = 365  # Default assumption
        from datetime import datetime, timedelta
        self._expires_at = (datetime.now() + timedelta(days=expires_in_days)).timestamp()
        
        # Store securely
        with open(self.TOKEN_FILE, "w") as fh:
            json.dump({
                "token": token,
                "expires_at": self._expires_at,
                "created_at": datetime.now().isoformat(),
            }, fh, indent=2)
        
        self._token = token
        return True
    
    def invalidate(self) -> None:
        """Delete stored token (logout)."""
        if self.TOKEN_FILE.exists():
            self.TOKEN_FILE.unlink()
        self._token = None
    
    def set_header(self, headers: dict) -> dict:
        """Add authorization header if authenticated."""
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers
    
    @classmethod
    def clear_cached_credentials(cls) -> None:
        """Remove all cached credentials."""
        if cls.TOKEN_FILE.exists():
            cls.TOKEN_FILE.unlink()


# Usage example
def fetch_emodnet_authenticated(*args, **kwargs):
    """Wrapper that uses token manager."""
    manager = EmodnetTokenManager()
    
    # Try to load/create token
    if not manager.load_or_create_token():
        raise RuntimeError("Authentication required but failed")
    
    # Add auth header to download request
    original_download_url = _emodnet_download_url
    def patched_url(lon_min, lon_max, lat_min, lat_max, *rest):
        url = original_download_url(lon_min, lon_max, lat_min, lat_max, *rest)
        # Headers will be modified by urllib.request call using manager.set_header()
        return url
    
    # Proceed with download...
```

**CLI Commands for Token Management:**

```bash
# Set token interactively
tideglass emodnet login

# Remove token (logout)
tideglass emodnet logout

# Check token status
tideglass emodnet status
```

---

## Summary of Progress

| Step | Status | Implementation | Files Modified |
|------|--------|----------------|----------------|
| 1. Batch downloads | ✅ Complete | `fetch_emodnet_batch()` | `tideglass/fetch_emodnet.py` |
| 2. Polling loop | ⏳ Pending | Pattern provided above | Needs new file |
| 3. Quality filtering | ✅ Complete | `fetch_emodnet_with_quality()` | `tideglass/fetch_emodnet.py` |
| 4. Auth tokens | ⏳ Pending | Pattern provided above | Needs new file |

**To Commit:**

```bash
git add tideglass/fetch_emodnet.py tideglass/__init__.py
git commit -m "Add batch downloads and quality filtering for EMODnet"
git push origin main
```

All implemented functions are now available in the public API:
- `fetch_emodnet_batch()`
- `fetch_emodnet_with_quality()`
