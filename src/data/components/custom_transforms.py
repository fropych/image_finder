from PIL import ImageFont, ImageDraw, Image
from numpy import random
import string
random.seed(42)

class RandomText:
    
    alphabet = list(string.ascii_letters)+[' ', '\n']
    def __init__(self, font) -> None:
        self.font = font
    
    def __call__(self, img: Image):
        img = img.copy()
        width, height = img.size

        font_size = int(width * random.uniform(0.03, 0.2))
        font = ImageFont.truetype(self.font, size=font_size)

        text_size = random.randint(20*int(1 - 3*font_size/width), int(width/font_size)*5)
        text = ''.join(random.choice(self.alphabet, text_size))
        text_color = random.choice(["white", "black"])
        stroke_color = "black" if text_color == "white" else "white"

        x = random.randint(
            0, int((0.5 + 2 * (0.2 - font_size / width)) * width)
        )
        y = random.randint(0, int(0.8 * height))
        
        img_draw = ImageDraw.Draw(img)
        img_draw.text(
            (x, y),
            text,
            fill=text_color,
            font=font,
            stroke_fill=stroke_color,
            stroke_width=int(font_size * 0.02) + 1,
        )
        return img

class RandomCircle:
    def __call__(self, img: Image):
        img = img.copy()
        width, height = img.size
        
        r1 = int(width * random.uniform(0.01, 0.35))
        r2 = int(width * random.uniform(0.01, 0.35))
        x = random.randint(width)
        y = random.randint(height)
        x1, x2 = x-r1, x
        y1, y2 = y, y+r2
        color = tuple(random.randint(0, 256, 3))

        img_draw = ImageDraw.Draw(img)
        img_draw.ellipse((x1,y1,x2,y2), fill=color, width=2, outline='black')
        return img
        