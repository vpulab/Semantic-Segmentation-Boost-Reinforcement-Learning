import numpy as np


class GridGenerator():

    def __init__(self, width=17, height=15):
        self.grid_w = width
        self.grid_h = height
        self.grid = np.array([[['[background]', 'Empty', 'Empty'] for a in np.arange(
            self.grid_w)] for b in np.arange(self.grid_h)], dtype='object')
        self.free_floor = np.arange(self.grid_w)
        self.havefloor = []
        self.floor_level = 12

    def GenerateClouds(self):
        # Rows 9,10 over the floor
        block_n = np.random.choice(np.arange(3), p=[0.1, 0.5, 0.4])
        # To not overlap hills and clouds
        #hill_av = np.arange(16)
        if block_n != 0:
            # where are they placed
            block_l = np.random.permutation(self.grid_w)[:block_n]

            for i in np.arange(block_n):
                # how many blocks stick together:
                size = np.random.choice(np.arange(3, 6), p=[0.4, 0.3, 0.3])
                # what type are the blocks. Bricks? or boxes?
                for k in np.arange(size):
                    # to place lengthy blocks around initial position
                    offset = np.power(-1, k)*((k+1)//2)
                    position = block_l[i]+offset
                    # if position is inside the grid, set the label
                    if position >= 0 and position < self.grid_w:
                        # np.delete(hill_av,position)
                        if k < size - 2:
                            self.grid[2, position, 0] = '[cloud_tm]'
                            self.grid[3, position, 0] = '[cloud_bm]'
                        else:
                            if offset < 0:
                                if 'cloud' in self.grid[2, position, 0]:
                                    self.grid[2, position, 0] = '[cloud_tm]'
                                    self.grid[3, position, 0] = '[cloud_bm]'
                                else:
                                    self.grid[2, position, 0] = '[cloud_tl]'
                                    self.grid[3, position, 0] = '[cloud_bl]'
                            else:
                                if 'cloud' in self.grid[2, position, 0]:
                                    self.grid[2, position, 0] = '[cloud_tm]'
                                    self.grid[3, position, 0] = '[cloud_bm]'
                                else:
                                    self.grid[2, position, 0] = '[cloud_tr]'
                                    self.grid[3, position, 0] = '[cloud_br]'

    def GenerateHills(self):
        # Generate hills.

        # Number of hills
        block_n = np.random.choice(np.arange(3), p=[0.1, 0.45, 0.45])
        if block_n != 0:
            block_h = np.random.choice([0, 1], size=block_n, p=[0.5, 0.5])
            # generate placements (doesnt take into account hill height)
            block_l = [5*i + x for i,
                       x in enumerate(sorted(np.random.randint(0, 7, size=3)))]

            free_floor_for_hills = self.free_floor
            for i in np.arange(block_n):
                # place tip
                self.grid[10+block_h[i], block_l[i], 0] = '[hill_t]'
                # middle row
                self.grid[11+block_h[i], block_l[i], 0] = '[hill_mm]'
                if block_l[i]-1 >= 0:
                    self.grid[11+block_h[i], block_l[i]-1, 0] = '[hill_ml]'
                    # bottom row, saves 1 comparison
                    self.grid[12+block_h[i], block_l[i]-1, 0] = '[hill_bl]'

                if block_l[i]+1 < self.grid_w:
                    self.grid[11+block_h[i], block_l[i]+1, 0] = '[hill_mr]'
                    # bottom row, saves 1 comparison
                    self.grid[12+block_h[i], block_l[i]+1, 0] = '[hill_br]'

                # The rest of the bottom row
                if block_l[i] - 2 >= 0:
                    self.grid[12+block_h[i], block_l[i]-2, 0] = '[hill_bll]'
                if block_l[i]+2 < self.grid_w:
                    self.grid[12+block_h[i], block_l[i]+2, 0] = '[hill_brr]'

                self.grid[12+block_h[i], block_l[i], 0] = '[hill_bm]'

                free_floor_for_hills = [x for x in free_floor_for_hills if x not in [
                    block_l[i]-2, block_l[i]-1, block_l[i], block_l[i]+1, block_l[i]+2]]

    def GenerateBushes(self):
        # Generates bushes (1 to 2 per image)
        block_n = np.random.choice(np.arange(1, 3), p=[0.6, 0.4])
        #print("Free floor for bushes: ",free_floor)
        if block_n != 0:
            # where are they placed
            block_l = np.random.permutation(self.free_floor)[:block_n]

            for i in np.arange(block_n):
                # how many blocks stick together:
                size = np.random.choice(np.arange(3, 6), p=[0.6, 0.3, 0.1])
                # what type are the blocks. Bricks? or boxes?
                for k in np.arange(size):
                    # to place lengthy blocks around initial position
                    offset = np.power(-1, k)*((k+1)//2)
                    position = block_l[i]+offset
                    # if position is inside the grid, set the label

                    if position >= 0 and position < self.grid_w:
                        # np.delete(hill_av,position)
                        if k < size - 2:
                            self.grid[12, position, 1] = '[bush_m]'
                        else:
                            if offset < 0:
                                if 'bush' in self.grid[12, position, 1]:
                                    self.grid[12, position, 1] = '[bush_m]'
                                else:
                                    self.grid[12, position, 1] = '[bush_l]'
                            else:
                                if 'bush' in self.grid[12, position, 1]:
                                    self.grid[12, position, 1] = '[bush_m]'
                                else:
                                    self.grid[12, position, 1] = '[bush_r]'

    def GenerateFloor(self):
        # Generate floor tiles
        # Last two rows will be floor, with random holes
        prob = 0.02
        holes = []
        for i in np.arange(self.grid_w):
            # randomly decide if there is a hole or not
            if np.random.uniform() <= prob:  # 2% of generating hole
                # if it generates a hole, its more probable that it has another one next to it
                if prob == 0.02:
                    prob = 0.85
                else:
                    prob = 0.02
                holes.append(i)
                continue
            else:  # no hole, fill with floor
                self.grid[13, i, 1] = '[floor]'
                self.grid[14, i, 1] = '[floor]'
                self.havefloor.append(i)

        for i in holes:
            self.free_floor = np.delete(
                self.free_floor, np.where(self.free_floor == i), axis=0)

    def GenerateTrees(self):
        # Get the number of available slots in the floor.
        trees = np.random.choice(np.arange(0, 4), p=[0.25, 0.45, 0.23, 0.07])

        if trees != 0:
            # 2 Types of trees, big top and small top. Both share trunk
            for tree in np.arange(trees):
                if len(self.havefloor) == 0:
                    print(
                        "There is no floor... Have you called GenerateFloor before placing trees? Placing them anywhere")
                    place = np.random.choice(np.arange(self.grid_w))
                else:
                    place = np.random.choice(self.havefloor)

                tree_type = np.random.choice(np.arange(0, 2))

                if tree_type == 0:  # trunk 2 top
                    self.grid[self.floor_level, place, 1] = '[tree_trunk]'
                    self.grid[self.floor_level-1, place, 1] = '[tree_tb]'
                    self.grid[self.floor_level-2, place, 1] = '[tree_tt]'
                else:
                    self.grid[self.floor_level, place, 1] = '[tree_trunk]'
                    self.grid[self.floor_level-1, place, 1] = '[tree_small]'

    def GenerateBlocks(self):
        # This function places unbrickable blocks
        # Must work similar to placing bricks
        max_blocks = 4
        block_number = np.random.choice(
            np.arange(0, max_blocks), p=[0.7, 0.20, 0.07, 0.03])
        #print("Free floor before blocks {}".format(self.free_floor))

        placed_block = []

        if block_number != 0:
            for block in np.arange(block_number):
                if len(self.havefloor) == 0:
                    print(
                        "There is no floor... Have you called GenerateFloor before generating blocks? Placing them anywhere")
                    place = np.random.choice(np.arange(self.grid_w))
                else:
                    place = np.random.choice(self.havefloor)

                block_group_height = np.random.choice(np.arange(1, 5))

                # First place the main block
                for k in np.arange(block_group_height):
                    self.grid[self.floor_level-k, place, 2] = '[block_hard]'

                placed_block.append(place)

                # Now, roll to check wether it grows or not and in which direction. The more blocks there are, less growth width.

                grow = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])

                if grow != 0:
                    grow_n = np.random.randint(0, max_blocks-block_number)
                    for block_strip in np.arange(grow_n):
                        loc = place + block_strip*grow

                        if loc > 0 and loc < self.grid_w:
                            height = np.random.choice(
                                np.arange(1, block_group_height+1))
                            for i in np.arange(height):
                                self.grid[self.floor_level-i,
                                          loc, 2] = '[block_hard]'

                            placed_block.append(loc)

            for i in placed_block:
                self.free_floor = np.delete(
                    self.free_floor, np.where(self.free_floor == i), axis=0)
            #print("Free floor after blocks {}".format(self.free_floor))

    def GeneratePipes(self):
        # Over those that have floor, pipes can appear, with low probability.
        pipes = np.random.choice(np.arange(0, 4), p=[0.7, 0.20, 0.07, 0.03])
        placed_pipes = []
        #print("Free floor before pipes {}".format(self.free_floor))
        if pipes != 0:
            for pipe in np.arange(pipes):
                # find where to place it (part of the pipe can be over a hole)
                if len(self.havefloor) == 0:
                    print(
                        "There is no floor... Have you called GenerateFloor before generating pipes? Placing them anywhere")
                    place = np.random.choice(np.arange(self.grid_w))
                else:
                    place = np.random.choice(self.havefloor)
                pipeH = np.random.choice(np.arange(2, 4))

                if place == 0:  # only the right part of the pipe will be visible
                    # if there is already a pipe there skip this one.
                    if ('pipe' in self.grid[12, place, 2]):
                        continue

                    self.grid[12, place, 2] = '[pipe_br]'

                    if pipeH == 2:
                        self.grid[11, place, 2] = '[pipe_tr]'
                    else:
                        self.grid[11, place, 2] = '[pipe_br]'
                        self.grid[10, place, 2] = '[pipe_tr]'

                    placed_pipes.append(place)
                    enemy = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])

                    if enemy > 0:
                        self.grid[12-pipeH, place,
                                  2] = '[piranha_'+str(enemy)+']'
                        self.grid[12-pipeH, place,
                                  2] = '[piranha_'+str(enemy)+']'
                else:
                    if ('pipe' in self.grid[12, place, 2]) or ('pipe' in self.grid[12, place-1, 2]):
                        continue

                    self.grid[12, place, 2] = '[pipe_br]'
                    self.grid[12, place-1, 2] = '[pipe_bl]'

                    if pipeH == 2:
                        self.grid[11, place, 2] = '[pipe_tr]'
                        self.grid[11, place-1, 2] = '[pipe_tl]'
                    else:
                        self.grid[11, place, 2] = '[pipe_br]'
                        self.grid[11, place-1, 2] = '[pipe_bl]'
                        self.grid[10, place, 2] = '[pipe_tr]'
                        self.grid[10, place-1, 2] = '[pipe_tl]'

                    placed_pipes.append(place-1)
                    placed_pipes.append(place)

                    # Does it have an enemy?
                    enemy = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
                    if enemy > 0:
                        self.grid[12-pipeH, place,
                                  2] = '[piranha_'+str(enemy)+']'
                        self.grid[12-pipeH, place-1,
                                  2] = '[piranha_'+str(enemy)+']'

        for i in placed_pipes:
            self.free_floor = np.delete(
                self.free_floor, np.where(self.free_floor == i), axis=0)
        #print("Free floor after pipes {}".format(self.free_floor))

    def PlaceBlocks(self, level=0):
        if level == 0:
            # At heights 4, 8 blocks may appear. If so, they appear in chunks of random size.
            # At height 4, higher probability of blocks of length 3-5.
            # Also, at height 4 different probabilities of 1 or more appearing.
            # HEIGHT 4
            # how many blocks?
            block_n = np.random.choice(np.arange(3), p=[0.15, 0.5, 0.35])
            if block_n != 0:
                # where are they placed
                block_l = np.random.permutation(self.grid_w)[:block_n]

                for i in np.arange(block_n):
                    # how many blocks stick together:
                    size = np.random.choice(
                        np.arange(1, 7), p=[0.05, 0.1, 0.23, 0.3, 0.22, 0.1])
                    # what type are the blocks. Bricks? or boxes?
                    b_type = np.random.choice(
                        np.arange(2), size=size, p=[0.8, 0.2])
                    for k in np.arange(size):
                        b_label = '[brick]'
                        if b_type[k] == 1:
                            b_label = '[box]'

                        # to place lengthy blocks around initial position
                        offset = np.power(-1, k)*((k+1)//2)
                        position = block_l[i]+offset
                        # if position is inside the grid, set the label
                        if position >= 0 and position < self.grid_w:
                            self.grid[9, position, 1] = b_label
            # Height 8
            # how many blocks?
            block_n = np.random.choice(np.arange(3), p=[0.6, 0.25, 0.15])
            if block_n != 0:
                # where are they placed
                block_l = np.random.permutation(self.grid_w)[:block_n]

                for i in np.arange(block_n):
                    # how many blocks stick together:
                    size = np.random.choice(
                        np.arange(1, 4), p=[0.65, 0.25, 0.1])
                    # what type are the blocks. Bricks? or boxes?
                    b_type = np.random.choice(
                        np.arange(2), size=size, p=[0.7, 0.3])
                    for k in np.arange(size):
                        b_label = '[brick]'
                        if b_type[k] == 1:
                            b_label = '[box]'

                        # to place lengthy blocks around initial position
                        offset = np.power(-1, k)*((k+1)//2)
                        position = block_l[i]+offset
                        # if position is inside the grid, set the label
                        if position >= 0 and position < self.grid_w:
                            self.grid[5, position, 1] = b_label

    def GenerateMushrooms(self):
        # place some mushrooms
        block_n = np.random.choice(np.arange(4), p=[0.1, 0.25, 0.35, 0.3])
        # print("Mushrooms:",block_n)
        if block_n != 0:
            # where are they placed
            random_free_floor = np.random.permutation(self.free_floor)
            block_l = random_free_floor[:block_n]
            self.free_floor = random_free_floor[block_n:]
            size = np.random.choice(np.arange(1, 4), p=[0.65, 0.25, 0.1])

            for i in np.arange(block_n):
                b_type = np.random.choice(
                    np.arange(2), size=size, p=[0.5, 0.5])
                for k in np.arange(size):
                    b_label = '[mush_1]'
                    if b_type[k] == 1:
                        b_label = '[mush_2]'

                    self.grid[12, block_l[i], 2] = b_label

    def GenerateKoopas(self):
        # Place koopas

        koopa_n = np.random.choice(np.arange(3), p=[0.8, 0.15, 0.05])
        # print("Mushrooms:",block_n)
        if koopa_n != 0:
            # where are they placed
            random_free_floor = np.random.permutation(self.free_floor)
            block_l = random_free_floor[:koopa_n]
            self.free_floor = random_free_floor[koopa_n:]
            size = np.random.choice(np.arange(1, 4), p=[0.65, 0.25, 0.1])

            for i in np.arange(koopa_n):
                b_type = np.random.choice(
                    np.arange(2), size=size, p=[0.5, 0.5])
                for k in np.arange(size):
                    b_label = '[koopa_1]'
                    if b_type[k] == 1:
                        b_label = '[koopa_2]'

                    self.grid[12, block_l[i], 2] = b_label

    def GenerateEnemies(self):
        self.GenerateMushrooms()
        self.GenerateKoopas()

    def GenerateMario(self):
        # at last, place mario. For this, first choose if he's on the floor or jumping
        mario_state = np.random.choice(['floor', 'jump'], p=[0.8, 0.2])

        if mario_state == 'floor':
            # can be one of four states
            mario_state = np.random.choice(['idle', 'walk1', 'walk2', 'walk3'], p=[
                                           0.25, 0.25, 0.25, 0.25])
            # has to be placed on a free slot in the floor
            label = '[mario_' + mario_state + ']'

            self.grid[12, self.free_floor[0], 2] = label
            # print(grid[12,free_floor[0],2])
        else:
            # if not, choose a random position in the air, check if its free or choose again.
            # there can be mario as high as the clouds (position 2 and 3) so from 11 to 2
            # have 9 x 17 slots. Do random number between 0 and 9x17 and select via modulo
            # CHECK AGAIN SEEMS LIKE NO MARIOS ON SECOND ROW
            placed = False
            while placed == False:
                position = np.random.randint(0, 9*17)
                x = position % 17
                y = position//17
                #print("Jump", position,"x", x,"y",y)
                if self.grid[2+y, x, 1] == 'Empty' and self.grid[2+y, x, 2] == 'Empty':
                    self.grid[2+y, x, 2] = '[mario_jump]'
                    placed = True

    def GenerateGrid(self, level):
        '''This function generates the grid with the elements that will be displayed in the image.
            Elements may vary depeding on the type of level being played, using a flag variable.

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
        # If level is type default
        if level[1] == '0' or level[1] == '1' or level[1] == 'O' or level[1] == 'I':
            self.GenerateClouds()

        # Common for all levels
        self.GenerateFloor()
        self.GenerateBlocks()
        self.PlaceBlocks()

        # Background decoration
        if level[1] == '0' or level[1] == 'O':
            self.GenerateHills()
            self.GenerateBushes()
        elif level[1] == '1' or level[1] == 'I':
            self.GenerateTrees()

        # Generate pipes
        self.GeneratePipes()

        # Generate enemies
        self.GenerateEnemies()

        # Place mario
        self.GenerateMario()

        return self.grid
