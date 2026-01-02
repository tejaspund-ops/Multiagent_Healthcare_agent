from datetime import datetime, timedelta, timezone
from jose import jwt

SECRET_KEY = "0112562871"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


if __name__ == "__main__":
    test_data = {"sub": "user_123"}
    token = create_access_token(test_data)
    print(f"Bearer {token}")
