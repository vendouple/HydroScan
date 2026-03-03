"""PlantNet API adapter for plant identification."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image


@dataclass(slots=True)
class PlantResult:
    """Single plant identification result."""
    scientific_name: str
    common_names: List[str]
    family: str
    genus: str
    score: float
    images: List[Dict[str, Any]]


@dataclass(slots=True)
class PlantNetResponse:
    """Full PlantNet API response."""
    query: Dict[str, Any]
    results: List[PlantResult]
    best_match: Optional[str]
    version: str
    remaining_requests: int


class PlantNetAdapter:
    """Adapter for PlantNet plant identification API."""
    
    API_URL = "https://my-api.plantnet.org/v2/identify/all"
    
    def __init__(self, api_key: str = "2b10eXbBu0Iz8dQdt5Olj9bv"):
        """Initialize PlantNet adapter with API key."""
        self.api_key = api_key
        self._session = requests.Session()
    
    def identify_from_file(
        self, 
        image_path: str | Path,
        organs: List[str] = None
    ) -> PlantNetResponse:
        """
        Identify plant from image file.
        
        Args:
            image_path: Path to image file
            organs: List of plant organs in image (flower, leaf, fruit, bark, auto)
                   Default is ["auto"]
        
        Returns:
            PlantNetResponse with identification results
        """
        if organs is None:
            organs = ["auto"]
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as f:
            return self._identify(f.read(), organs)
    
    def identify_from_bytes(
        self, 
        image_data: bytes,
        organs: List[str] = None
    ) -> PlantNetResponse:
        """
        Identify plant from image bytes.
        
        Args:
            image_data: Raw image bytes
            organs: List of plant organs in image
        
        Returns:
            PlantNetResponse with identification results
        """
        if organs is None:
            organs = ["auto"]
        
        return self._identify(image_data, organs)
    
    def identify_from_pil(
        self, 
        image: Image.Image,
        organs: List[str] = None,
        format: str = "JPEG"
    ) -> PlantNetResponse:
        """
        Identify plant from PIL Image.
        
        Args:
            image: PIL Image object
            organs: List of plant organs in image
            format: Image format for encoding (JPEG, PNG)
        
        Returns:
            PlantNetResponse with identification results
        """
        if organs is None:
            organs = ["auto"]
        
        buffer = BytesIO()
        # Convert RGBA to RGB for JPEG
        if format.upper() == "JPEG" and image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffer, format=format)
        return self._identify(buffer.getvalue(), organs)
    
    def _identify(
        self, 
        image_data: bytes, 
        organs: List[str]
    ) -> PlantNetResponse:
        """
        Internal method to call PlantNet API.
        
        Args:
            image_data: Raw image bytes
            organs: List of plant organs
        
        Returns:
            PlantNetResponse with identification results
        """
        params = {
            "api-key": self.api_key,
            "include-related-images": "true",
        }
        
        # Build multipart form data
        files = [("images", ("image.jpg", image_data, "image/jpeg"))]
        data = [("organs", organ) for organ in organs]
        
        try:
            response = self._session.post(
                self.API_URL,
                params=params,
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            return self._parse_response(response)
        except requests.exceptions.RequestException as e:
            raise PlantNetError(f"API request failed: {e}") from e
    
    def _parse_response(self, response: requests.Response) -> PlantNetResponse:
        """Parse API response into structured data."""
        data = response.json()
        
        results = []
        for result in data.get("results", []):
            species = result.get("species", {})
            results.append(PlantResult(
                scientific_name=species.get("scientificNameWithoutAuthor", "Unknown"),
                common_names=species.get("commonNames", []),
                family=species.get("family", {}).get("scientificNameWithoutAuthor", "Unknown"),
                genus=species.get("genus", {}).get("scientificNameWithoutAuthor", "Unknown"),
                score=result.get("score", 0.0),
                images=result.get("images", [])
            ))
        
        # Get remaining API requests from headers
        remaining = int(response.headers.get("X-Rate-Limit-Remaining", -1))
        
        return PlantNetResponse(
            query=data.get("query", {}),
            results=results,
            best_match=data.get("bestMatch", None),
            version=data.get("version", "unknown"),
            remaining_requests=remaining
        )


class PlantNetError(Exception):
    """Exception raised for PlantNet API errors."""
    pass
