from functools import cache, singledispatch
from pathlib import Path

import numpy as np
from PIL import Image

from giftools import save_transparent_gif

gifs = {
    # (197, 17, 17): "amogus01.gif",  # red
    (19, 46, 209): "amogus02.gif",  # blue
    # (17, 127, 45): "amogus03.gif",  # green
    # (237, 84, 186): "amogus04.gif",  # pink
    # (239, 125, 13): "amogus05.gif",  # orange
    (245, 245, 87): "amogus06.gif",  # yellow
    (63, 71, 78): "amogus07.gif",  # black
    (214, 224, 240): "amogus08.gif",  # white
    # (107, 47, 187): "amogus09.gif",  # purple
    # (113, 73, 30): "amogus10.gif",  # brown
    # (56, 254, 220): "amogus11.gif",  # cyan
    # (80, 239, 57): "amogus12.gif",  # lime
}


def load_gif(path):
    img = Image.open(Path("resources", path))
    img.convert("RGBA")
    return img


loaded_gifs = {v: load_gif(v) for v in gifs.values()}


@singledispatch
def get_closest_colour(arg):
    print(type(arg))
    raise NotImplemented


@get_closest_colour.register
@cache
def _(arg: tuple):
    r, g, b = arg
    distances = {
        k: np.linalg.norm((r - k[0], g - k[1], b - k[2]))
        for k in gifs
    }
    return gifs[min(distances, key=distances.get)]


@get_closest_colour.register
def _(arg: np.ndarray):
    tup = arg[0], arg[1], arg[2]
    return get_closest_colour(tup)


def main():
    input_image = Image.open("me.png")
    arr = np.array(input_image)
    # y, x, 3 or 4
    shape = arr.shape
    tile_size = np.array((63, 74))
    # y, x
    number_tiles = np.ceil(shape[:2] / tile_size).astype(np.int32)
    # x, y
    number_tiles = np.roll(number_tiles, 1)
    # Expects width, height
    scaled_down = input_image.resize(number_tiles, resample=Image.LANCZOS)
    arr = np.array(scaled_down)
    # y, x, 3 or 4
    shape = arr.shape
    yrange = np.arange(shape[0])
    xrange = np.arange(shape[1])
    ys, xs = np.meshgrid(yrange, xrange)
    phase = (xs - ys) % 6
    amogus = arr.astype(dtype="object")
    amogus = np.apply_along_axis(get_closest_colour, 2, amogus)
    output_size = tuple(number_tiles * np.roll(tile_size, 1))
    frames = [Image.new("RGBA", output_size, (0, 0, 0, 0)) for _ in range(6)]
    for offset, frame in enumerate(frames):
        for y in range(0, shape[0]):
            for x in range(0, shape[1]):
                f = (offset + phase[x, y]) % 6
                g = amogus[y, x]
                gif = loaded_gifs[g]
                gif.seek(f)
                coords = tuple(np.roll(tile_size * (y, x), 1))
                frame.paste(gif, coords)

    save_transparent_gif(frames, 50, "output.gif")


if __name__ == "__main__":
    main()
