from pathlib import Path

__all__ = ["IMG_CITY", "IMG_CROWN", "IMG_MOUNTAIN", "IMG_OBSTACLE", "FONT"]

DIR = Path(__file__).parent
IMG_CITY = DIR / "city.png"
IMG_CROWN = DIR / "crown.png"
IMG_MOUNTAIN = DIR / "mountain.png"
IMG_OBSTACLE = DIR / "obstacle.png"
FONT = DIR / "Quicksand-Bold.ttf"
