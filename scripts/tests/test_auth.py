#!/usr/bin/env python3
"""
Test authentication logic directly
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.database.dependencies import get_db
from backend.database.db_utils import get_user_by_username
from backend.core.security import verify_password

def test_auth():
    db = next(get_db())
    try:
        # Test getting user
        user = get_user_by_username(db, "officer")
        if not user:
            print("❌ User 'officer' not found")
            return

        print(f"✅ Found user: {user.username}, role: {user.role}, disabled: {user.disabled}")
        print(f"Hashed password: {user.hashed_password}")

        # Test password verification
        test_password = "password"
        is_valid = verify_password(test_password, user.hashed_password)
        print(f"Password verification for '{test_password}': {is_valid}")

        if is_valid:
            print("✅ Authentication should work!")
        else:
            print("❌ Password verification failed")

    finally:
        db.close()

if __name__ == "__main__":
    test_auth()