from copy import copy


is_in_the_same_row = lambda ya, yb: (abs(ya -yb) <= 20)


def join_separated(bands):
    new_bands = []

    for y0a, y1a, x0a, x1a in bands:
        one_row_bands = []

        for y0b, y1b, x0b, x1b in bands:
            if (is_in_the_same_row(y0a, y0b)) and (is_in_the_same_row(y1a, y1b)):
                one_row_bands.append((y0b, y1b, x0b, x1b))

        y0 = min([band[0] for band in one_row_bands])
        y1 = max([band[1] for band in one_row_bands])
        x0 = min([band[2] for band in one_row_bands])
        x1 = max([band[3] for band in one_row_bands])

        new_bands.append((y0, y1, x0, x1))

    return list(set(new_bands))


if __name__ == '__main__':
    bands = [(220, 252, 436, 487), (220, 252, 487, 976), (42, 59, 171, 225), (255, 282, 42, 75)]

    print(join_separated(bands))