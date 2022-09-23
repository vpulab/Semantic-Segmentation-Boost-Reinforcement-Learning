import os
import shutil
import numpy as np
import argparse
from cv2 import cv2


'''
This script is used to extract the single sprites included in the tileset.png in the same folder.
'''
def ExtractTiles(path):
    # Load the tileset
    tileset = cv2.imread(path)[..., ::-1]

    y, x, z = tileset.shape  # y,x,z

    # the image includes 5 levels, from which only 4 are from the original super mario
    # (Overworld, Underwold,Underwater and Castle)

    # Iterate over each level region and extract sprites.
    # Each region has 9 x 16 (hxw) sprites, with the 6 in the bottom right corner being 2x as high.

    # Width and height for each level set of tiles
    lvl_wide = x//3
    lvl_height = y//2

    # size for each tile
    grid_size = (16, 16)  # y,x

    # Extract all sprites for all valid levels
    #(Overworld, Underwold,Underwater and Castle)
    for level in np.arange(4):
        # offsets for each level
        x_offset_lvl = 1 + (lvl_wide+1)*(level % 3)
        y_offset_lvl = 12 + (level//3)*(37+136)

        # extract per row
        for y_i in np.arange(136//grid_size[0]):
            # extract per column
            y_offset = (grid_size[0]+1)*y_i + y_offset_lvl
            for x_i in np.arange((x//3)//grid_size[1]-1):
                x_offset = (grid_size[1]+1)*x_i + x_offset_lvl

                # if row is 6 and column is 10 or bigger, skip as those are 2x height sprites
                # probably resized on input tho
                if (y_i == 6 and x_i > 9):
                    continue
                # Get the 2x height sprites
                elif (y_i == 7 and x_i > 9):
                    sprite = tileset[y_offset-grid_size[0]-1:y_offset +
                                    grid_size[0]-1, x_offset:x_offset+grid_size[0], :]

                else:
                    sprite = tileset[y_offset:y_offset+grid_size[0],
                                    x_offset:x_offset+grid_size[0], :]

                sprite_write = cv2.cvtColor(sprite, cv2.COLOR_RGB2BGR)
                cv2.imwrite("Sprites/Sprite%d_%d_%d.png" %
                            (level, x_i, y_i), sprite_write)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Extract tiles from tileset.")

    parser.add_argument('--path_tileset', '-t',
                        type=str,
                        default="tilesets/tileset.png",
                        required=False,
                        help='Path to tileset file.',
                        )

    FLAGS, _ = parser.parse_known_args()

    dir = 'Sprites'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    #Cut sprites from tileset
    ExtractTiles(FLAGS.path_tileset)