from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from typing import Optional

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from request headers"""
    api_key = credentials.credentials
    expected_key = os.getenv("API_KEY")
    
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

def get_api_key_header():
    """Return API key header name for documentation"""
    return "Authorization: Bearer YOUR_API_KEY"

