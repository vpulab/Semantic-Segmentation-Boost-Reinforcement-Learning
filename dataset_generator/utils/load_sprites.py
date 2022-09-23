from cv2 import cv2
import numpy as np

'''
This file stores all loaders for all the supported sprites. All loaders receive a parameter to properly load
correct sprite based on appearance.
'''
class SpriteLoader():
    ''' Class with different load functtions which return a dictionary with pairs
        of <str:label,cv2.mat: sprite>
    '''

    ###ENVIRONMENT LOADERS

    def loadFloor(level=0):
        level = str(level)
        floor =  cv2.imread("Sprites/Sprite"+level+"_0_7.png")[...,::-1]
        return floor

    def loadBox(level=0):
        level = str(level)
        box = cv2.imread("Sprites/Sprite"+level+"_0_3.png")[...,::-1]
        return box

    def loadBrick(level=0):
        level = str(level)
        box = cv2.imread("Sprites/Sprite"+level+"_1_5.png")[...,::-1]
        return box

    def loadBlock(level=0):
        level = str(level)
        block = cv2.imread("Sprites/Sprite"+level+"_5_3.png")[...,::-1]
        return block

    def loadClouds(level=0):
        #Convert to str to be able to concatenate it
        level = str(level)

        #Read clouds sprites
        clouds = {'[cloud_tl]':cv2.imread("Sprites/Sprite"+level+"_13_3.png")[...,::-1],
                    '[cloud_bl]':cv2.imread("Sprites/Sprite"+level+"_13_4.png")[...,::-1],
                    '[cloud_tm]':cv2.imread("Sprites/Sprite"+level+"_14_3.png")[...,::-1],
                    '[cloud_bm]':cv2.imread("Sprites/Sprite"+level+"_14_4.png")[...,::-1],
                    '[cloud_tr]':cv2.imread("Sprites/Sprite"+level+"_15_3.png")[...,::-1],
                    '[cloud_br]':cv2.imread("Sprites/Sprite"+level+"_15_4.png")[...,::-1]
                    }
        return clouds

    def loadBushes(level=0):

        #Convert to str to be able to concatenate it.
        level = str(level)
        #Read bushes sprites
        bushes = {'[bush_l]':cv2.imread("Sprites/Sprite"+level+"_13_5.png")[...,::-1],
                        '[bush_m]':cv2.imread("Sprites/Sprite"+level+"_14_5.png")[...,::-1],
                        '[bush_r]':cv2.imread("Sprites/Sprite"+level+"_15_5.png")[...,::-1],
                        }
        return bushes

    def loadHills(level=0):

        #Convert to str to be able to concatenate it.
        level = str(level)
        #Load hill sprites
        hill     =   {'[hill_t]':cv2.imread("Sprites/Sprite"+level+"_10_3.png")[...,::-1],
                       '[hill_ml]':cv2.imread("Sprites/Sprite"+level+"_9_4.png")[...,::-1],
                       '[hill_mm]':cv2.imread("Sprites/Sprite"+level+"_10_4.png")[...,::-1],
                       '[hill_mr]':cv2.imread("Sprites/Sprite"+level+"_11_4.png")[...,::-1],
                       '[hill_bll]':cv2.imread("Sprites/Sprite"+level+"_8_5.png")[...,::-1],
                       '[hill_bl]':cv2.imread("Sprites/Sprite"+level+"_9_5.png")[...,::-1],
                       '[hill_bm]':cv2.imread("Sprites/Sprite"+level+"_10_5.png")[...,::-1],
                       '[hill_br]':cv2.imread("Sprites/Sprite"+level+"_11_5.png")[...,::-1],
                       '[hill_brr]':cv2.imread("Sprites/Sprite"+level+"_12_5.png")[...,::-1]
                        }

        return hill

    def loadTrees(level=0):
        #to str for concatenation
        level = str(level)
        #load tree sprites
        tree = {'[tree_trunk]':cv2.imread("Sprites/Sprite"+level+"_14_2.png")[...,::-1],
                '[tree_tb]':cv2.imread("Sprites/Sprite"+level+"_14_1.png")[...,::-1],
                '[tree_tt]':cv2.imread("Sprites/Sprite"+level+"_14_0.png")[...,::-1],
                '[tree_small]':cv2.imread("Sprites/Sprite"+level+"_13_1.png")[...,::-1]
                }
        return tree



    def loadPipes(level=0):
        #Convert to str to be able to concatenate it.
        level = str(level)
        #Load sprites
        pipe = {'[pipe_tl]':cv2.imread("Sprites/Sprite"+level+"_6_2.png")[...,::-1],
                '[pipe_tr]':cv2.imread("Sprites/Sprite"+level+"_7_2.png")[...,::-1],
                '[pipe_bl]':cv2.imread("Sprites/Sprite"+level+"_6_3.png")[...,::-1],
                '[pipe_br]':cv2.imread("Sprites/Sprite"+level+"_7_3.png")[...,::-1]
            }
        return pipe

    #MARIO LOADER
    def loadMario():
        mario = {'[mario_idle]':cv2.imread("MarioSprites/idle.png")[...,::-1],
                      '[mario_walk1]':cv2.imread("MarioSprites/walk1.png")[...,::-1],
                      '[mario_walk2]':cv2.imread("MarioSprites/walk2.png")[...,::-1],
                      '[mario_walk3]':cv2.imread("MarioSprites/walk3.png")[...,::-1],
                      '[mario_jump]':cv2.imread("MarioSprites/jump.png")[...,::-1]
                      }
        return mario
    
    #ENEMY LOADERS
    def loadGoombas(tileset = 0 ):
        mushroom = {'[mush_1]':cv2.imread("EnemySprites/mushroom_"+str(tileset)+"_0.png")[...,::-1],
                    '[mush_2]':cv2.imread("EnemySprites/mushroom_"+str(tileset)+"_1.png")[...,::-1]}
        return mushroom
    
    def loadKoopa(tileset = 0):
        koopa = {'[koopa_1]':cv2.imread("EnemySprites/koopa_"+str(tileset)+"_0.png")[...,::-1],
                 '[koopa_2]':cv2.imread("EnemySprites/koopa_"+str(tileset)+"_1.png")[...,::-1]}
        
        return koopa

    def loadPiranha(tileset = 0):
        piranha = {'[piranha_1]':cv2.imread("EnemySprites/piranha_"+str(tileset)+"_0.png")[...,::-1],
                   '[piranha_2]':cv2.imread("EnemySprites/piranha_"+str(tileset)+"_1.png")[...,::-1]}
        
        return piranha


    #GENERATE SEGMENTATION GT FOR SOME SPRITES

    def GenerateSSGT(object,class_color, no_label_color = [0,0,0], background_color= np.array([147,187,236])):
        #First create dictionary to hold images and keys
        ssgt = {}

        #Iterate over already loaded object
        for key, value in object.items():
                #copy the sprite
                ssgt[key] = value.copy()

                #modify the sprite
                for i in np.arange(value.shape[0]):
                    for j in np.arange(value.shape[1]):                
                        if (value[i,j,:] == background_color).all():
                            ssgt[key][i,j,:] = no_label_color
                        else:
                            ssgt[key][i,j,:] = class_color

        return ssgt
    def SpriteSSGT(frame,class_color, no_label_color = [0,0,0], background_color= np.array([147,187,236])):

        ssf = frame.copy()

        for i in np.arange(frame.shape[0]):
                    for j in np.arange(frame.shape[1]):                
                        if (frame[i,j,:] == background_color).all():
                            ssf[i,j,:] = no_label_color
                        else:
                            ssf[i,j,:] = class_color

        return ssf
