# %%

import os
import shutil

import multiprocessing
import pandas as pd
import numpy as np
from cv2 import cv2
import time

import argparse
from tqdm import tqdm

from utils.generate_grid import GridGenerator
from utils.load_sprites import SpriteLoader
# the background color changes based on level


class FrameGenerator():
    ''' Super mario frames have a dimension of
        256 x 240 x 3 (x,y,c)
        The idea is to place labels inside a grid
        and then fill with the corresponding image and
        its segmentation
        '''

    def __init__(self, sprite_dataset="label_assignment/sprite_labels_correspondence.csv", cores=1):
        '''Initializes the frame generator'''
        # For faster generation
        self.cores = cores
        # load all sprites (routes of)
        self.sprites = pd.read_csv(sprite_dataset, sep=',')
        # available classes
        self.labels = self.sprites.Label.unique()
        # background color (for level 1)

        self.background_colors = {0: np.array(
            [147, 187, 236]), 1: np.array([0, 0, 0])}
        self.sprite_bg_color = np.array([147, 187, 236])
        # load all sprites and semantically segment them

        # Set size for the label grid, in terms of squared sprite blocks.
        # Set to 15 and 17 to do random crops on training to get horizontal "displacement"
        # to avoid "grid"-y look
        self.grid_h = 15
        self.grid_w = 17

        # Set existing classes
        self.classes = {"default": 0,
                        'floor': 1,
                        'brick': 2,
                        'box': 3,
                        'enemy': 4,
                        'mario': 5}

        # For segmented frames, set a color per class.
        self.classcolors = {"default": [0, 0, 0],
                            'floor': [0, 0, 255],
                            'brick': [127, 127, 0],
                            'box': [0, 255, 0],
                            'enemy': [255, 0, 0],
                            'mario': [255, 255, 0]}

    def SetLevelSprites(self, level):
        '''This function loads sprites and textures that will be used in the image.
            Level = 'xyz' with:
            x - Level tileset to use (not relevant for grid generation)
            y - type of level:
                - 0 means default level, with bushes and hills in the background
                - 1 means default level with trees in the background
                - 2 means underground level. No trees or bushes in the background
                - 3 means castle level (not implemented)
                - 4 means mushroom level (not implemented)
                - O means default level with alternate bushes
                - I means default level with alternate trees

            z - background color (not relevant for grid generation)
                - 0 default blue
                - 1 black 

        '''
        # Parameter is tileset to choose from.
        tileset = int(level[0])

        if level[1] == '2':
            tileset = 1

       # For background elements
        self.floor = SpriteLoader.loadFloor(tileset)
        self.box = SpriteLoader.loadBox(tileset)
        self.brick = SpriteLoader.loadBrick(tileset)
        self.pipe = SpriteLoader.loadPipes(tileset)
        self.block = SpriteLoader.loadBlock(tileset)

        if level[1] == '0' or level[1] == '1':
            tileset = 0
        elif level[1] == 'O' or level[1] == 'I':
            tileset = 3
        else:
            tileset = 0
        # For background elements
        self.hill = SpriteLoader.loadHills(tileset)
        self.clouds = SpriteLoader.loadClouds(0)
        self.bushes = SpriteLoader.loadBushes(tileset)
        self.trees = SpriteLoader.loadTrees(tileset)

        # Generate segmentation gt for level elements
        self.spipe = SpriteLoader.GenerateSSGT(
            self.pipe, self.classcolors['floor'])
        self.seg_floor = SpriteLoader.SpriteSSGT(
            self.floor, self.classcolors['floor'])
        self.sbox = SpriteLoader.SpriteSSGT(self.box, self.classcolors['box'])
        self.sbrick = SpriteLoader.SpriteSSGT(
            self.brick, self.classcolors['brick'])
        self.sblock = SpriteLoader.SpriteSSGT(
            self.brick, self.classcolors['floor'])

    def LoadSprites(self, level):
        '''Load sprites for mario, enemies and generates their ground truth.'''
        # cave sprites
        if level[1] == '2':
            tileset = 1
        else:
            tileset = 0

        # Load Enemies
        self.mushroom = SpriteLoader.loadGoombas(tileset)
        self.smushroom = SpriteLoader.GenerateSSGT(
            self.mushroom, self.classcolors['enemy'])
        self.koopa = SpriteLoader.loadKoopa(tileset)
        self.skoopa = SpriteLoader.GenerateSSGT(
            self.koopa, self.classcolors['enemy'])
        self.piranha = SpriteLoader.loadPiranha(tileset)
        self.spiranha = SpriteLoader.GenerateSSGT(
            self.piranha, self.classcolors['enemy'])

        # Load mario sprites
        self.mario = SpriteLoader.loadMario()
        self.smario = SpriteLoader.GenerateSSGT(
            self.mario, self.classcolors['mario'])

    def generate_frame(self, level='000', grid=[]):
        # Generate the grid for the level
        if grid == []:
            grid = self.generate_grid(level)

        self.SetLevelSprites(level)

        self.LoadSprites(level)

        if level[1] == '2':
            bg_color = 1
        else:
            bg_color = int(level[2])

        self.background_color = self.background_colors[bg_color]
        frame = np.zeros((16*15, 16*self.grid_w, 3))
        frame = frame + self.background_color
        sframe = np.zeros((16*15, 16*self.grid_w, 3))
        classframe = np.zeros((16*15, 16*self.grid_w))

        missing_right = False  # for piranha

        # Generate frame and semantically segmented frame
        for row in np.arange(grid.shape[0]):
            frow = row * 16  # index iterator
            for column in np.arange(grid.shape[1]):
                fcol = column*16  # index iterator
                # Print tile
                # First print background for the tile
                if grid[row, column, 0] == '[background]':
                    # Iterate over pixels of the grid
                    for i in np.arange(16):
                        for j in np.arange(16):
                            frame[frow+i, fcol+j, :] = self.background_color
                            sframe[frow+i, fcol+j] = [0, 0, 0]
                elif grid[row, column, 0][1:6] == 'cloud':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.clouds[grid[row, column, 0]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.clouds[grid[row, column, 0]][i, j]
                elif grid[row, column, 0][1:5] == 'hill':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.hill[grid[row, column, 0]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.hill[grid[row, column, 0]][i, j]
                else:
                    for i in np.arange(16):
                        for j in np.arange(16):
                            frame[frow+i, fcol+j] = [255, 255, 255]
                            sframe[frow+i, fcol+j] = [255, 255, 255]

                # Paint "mid depth"
                # paint floor
                if grid[row, column, 1] == '[floor]':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            frame[frow+i, fcol+j, :] = self.floor[i, j]
                            sframe[frow+i, fcol+j, :] = self.seg_floor[i, j]
                            classframe[frow+i, fcol+j] = self.classes['floor']
                elif grid[row, column, 1] == '[box]':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.box[i, j, :] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j, :] = self.box[i, j]
                                sframe[frow+i, fcol+j, :] = self.sbox[i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['box']

                elif grid[row, column, 1] == '[brick]':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.brick[i, j, :] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j, :] = self.brick[i, j]
                                sframe[frow+i, fcol+j, :] = self.sbrick[i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['brick']

                elif grid[row, column, 1][1:5] == 'bush':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.bushes[grid[row, column, 1]][i, j, :] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.bushes[grid[row, column, 1]][i, j]

                elif grid[row, column, 1][1:5] == 'tree':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            if (self.trees[grid[row, column, 1]][i, j, :] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.trees[grid[row, column, 1]][i, j]

                # Print characters
                if grid[row, column, 2][1:5] == 'mush':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            # Only print non background pixels
                            if (self.mushroom[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.mushroom[grid[row, column, 2]][i, j]
                                sframe[frow+i, fcol+j,
                                       :] = self.smushroom[grid[row, column, 2]][i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['enemy']

                if grid[row, column, 2][1:6] == 'koopa':
                    for i in np.arange(32):
                        for j in np.arange(16):
                            row_off = frow+i-16
                            # Only print non background pixels
                            if (self.koopa[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                frame[row_off, fcol+j,
                                      :] = self.koopa[grid[row, column, 2]][i, j]
                                sframe[row_off, fcol+j,
                                       :] = self.skoopa[grid[row, column, 2]][i, j]
                                classframe[row_off, fcol +
                                           j] = self.classes['enemy']

                if grid[row, column, 2][1:8] == 'piranha':
                    # Columns go from 0 to x, increasing, so first reaches left side of a "piranha cell"
                    # Special case is for column 0 where it could be a right side so it should offset left.
                    if missing_right == False:
                        y = np.random.randint(1, 21)

                        piranha_height = np.random.choice([0, y])

                    if grid[row+1, column, 2] == '[pipe_tl]':  # then only print half piranha

                        for i in np.arange(32):
                            for j in np.arange(8):
                                row_off = frow+i-16 + piranha_height
                                col_off = fcol+j + 8
                                if (self.piranha[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                    frame[row_off, col_off,
                                          :] = self.piranha[grid[row, column, 2]][i, j]
                                    sframe[row_off, col_off,
                                           :] = self.spiranha[grid[row, column, 2]][i, j]
                                    classframe[row_off,
                                               col_off] = self.classes['enemy']
                        missing_right = True

                    if grid[row+1, column, 2] == '[pipe_tr]':  # then only print half piranha

                        for i in np.arange(32):
                            for j in np.arange(8):
                                row_off = frow+i-16 + piranha_height
                                col_off = fcol+j

                                if (self.piranha[grid[row, column, 2]][i, j+8] != self.sprite_bg_color).any():
                                    frame[row_off, col_off,
                                          :] = self.piranha[grid[row, column, 2]][i, j+8]
                                    sframe[row_off, col_off,
                                           :] = self.spiranha[grid[row, column, 2]][i, j+8]
                                    classframe[row_off,
                                               col_off] = self.classes['enemy']
                        missing_right = False

                if grid[row, column, 2][1:5] == 'pipe':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            # Only print non background pixels
                            if (self.pipe[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.pipe[grid[row, column, 2]][i, j]
                                #print("Saved color",frame[frow+i,fcol+j])
                                sframe[frow+i, fcol+j,
                                       :] = self.spipe[grid[row, column, 2]][i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['floor']

                if grid[row, column, 2][1:6] == 'block':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            # Only print non background pixels
                            if (self.block[i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j, :] = self.block[i, j]
                                #print("Saved color",frame[frow+i,fcol+j])
                                sframe[frow+i, fcol+j, :] = self.sblock[i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['floor']

                # Print characters
                if grid[row, column, 2][1:6] == 'mario':
                    for i in np.arange(16):
                        for j in np.arange(16):
                            # print("Color",self.floor[i,j])
                            # Only print non background pixels
                            if (self.mario[grid[row, column, 2]][i, j] != self.sprite_bg_color).any():
                                frame[frow+i, fcol+j,
                                      :] = self.mario[grid[row, column, 2]][i, j]
                                #print("Saved color",frame[frow+i,fcol+j])
                                sframe[frow+i, fcol+j,
                                       :] = self.smario[grid[row, column, 2]][i, j]
                                classframe[frow+i, fcol +
                                           j] = self.classes['mario']

        return frame, sframe, classframe

    def generate_grid(self, level):

        grid_gen = GridGenerator()

        return grid_gen.GenerateGrid(level)

    def GenerateSamples(self, init_filenumber, end_filenumber, seed, w_tqdm=False):
        # This function generates frames and their label and semantic segmentation ground truths.
        np.random.seed(seed)

        files = None
        if w_tqdm == True:
            files = tqdm(np.arange(init_filenumber, end_filenumber))
        else:
            files = np.arange(init_filenumber, end_filenumber)
        for i in files:
            x = '0'  # np.random.choice(['0','1'])
            y = np.random.choice(['0', '1', '2', 'O', 'I'])
            z = np.random.choice(['0', '1'], p=[.8, .2])
            level = x + y + z
            #level = np.random.choice([0,1])
            frame, sframe, classframe = self.generate_frame(level)
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite("dataset/PNG/%d.png" % (i), frame)
            sframe = cv2.cvtColor(sframe.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite("dataset/Segmentation/%d.png" % (i), sframe)
            # labels = framegen.GenerateLabelImageFromSegmentation(sframe) #Esto habra que cambiarlo para que segun genere la segmentacion lo haga
            cv2.imwrite("dataset/Labels/%s.png" % (i), classframe)
        # image_list.write(str(filename))

    def GenerateDataset(self, samples):
        '''Generates a dataset of a given size.'''
        start = time.time()
        # gets number of available threads
        threads = self.cores
        # generates different random seeds for each thread to avoid repetitions in the dataset
        seeds = np.random.randint(100, size=threads)

        #creates the folder (removes if previously exists)
        dir = 'dataset'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        #and subfolders
        os.makedirs('dataset/PNG/')
        os.makedirs('dataset/Segmentation/')
        os.makedirs('dataset/Labels/')

        # If only uses one core, execute the function once
        if self.cores == 1:
            level = np.random.choice([0, 1])

            self.GenerateSamples(0, samples, level, w_tqdm=True)
        else:
            # otherwise, distribute amount of samples between threads.
            step = samples//self.cores

            ppool = multiprocessing.Pool(threads)
            ranges = step*np.arange(self.cores+1)

            ppool.starmap(self.GenerateSamples, zip(
                ranges[:-1], ranges[1:], seeds))

        end = time.time()

        print("Elapsed time:", end-start)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Generate a semantic segmentation dataset with synthetic super mario frames")

    parser.add_argument('--cores', '-c',
                        type=int,
                        default=1,
                        required=False,
                        help='How many cores to use, speeds up generation. Set to 1 if using windows',
                        )

    parser.add_argument('--samples', '-s',
                        type=int,
                        default=0,
                        required=True,
                        help='How many images to generate.',
                        )

    FLAGS, _ = parser.parse_known_args()

    # Generate the dataset
    framegen = FrameGenerator(cores=FLAGS.cores)
    framegen.GenerateDataset(FLAGS.samples)
