import pygame
import os

class Tileset:

    # this class defines a tileset. It reads a .png style tileset and cut it into small pieces
    # @file is the file_dir of the tileset image, @size is the pixel size per tile, @margin & @spacing are the distance beween tiles and border

    def __init__(self, file, size=(16, 16), margin=0, spacing=0):
        #print(file)
        self.file = file
        self.size = size
        self.margin = margin
        self.spacing = spacing

        if file[-3:]=="png":
            self.image = pygame.image.load(file)
            self.rect = self.image.get_rect()
        self.tiles = []
        self.load()

    # Load the tile. Return a one dim tile list.
    def load(self):

        if self.file[-3:]=="png":

            self.tiles = []
            x0 = y0 = self.margin
            w, h = self.rect.size
            dx = self.size[0] + self.spacing
            dy = self.size[1] + self.spacing
            for x in range(x0, w, dx):
                for y in range(y0, h, dy):
                    tile = pygame.Surface(self.size)
                    #tile.set_colorkey((0,0,0))
                    tile.blit(self.image, (0, 0), (x, y, *self.size))
                    self.tiles.append(tile)

        else:
            dirs = os.listdir(self.file)
            num=[]
            for _dir in dirs: 
                num.append(int(_dir[:-4]))
            num.sort()
            for i in num:
                tile = pygame.image.load(self.file+"\\"+str(i)+".png")
                self.tiles.append(tile)

    def __str__(self):
        return f'{self.__class__.__name__} file:{self.file} tile:{self.size}'
