import random
from PIL import Image, ImageDraw, ImageFont
import requests
import sys
import urllib
from pprint import pprint


image_info = ['/Users/chetandev/Desktop/swati/DSC00919.jpg',
              '/Users/chetandev/Desktop/swati/DSC01084.jpg',
               '/Users/chetandev/Desktop/swati/IMG_0398.jpg',
              '/Users/chetandev/Desktop/swati/DSC01299.jpg',
              '/Users/chetandev/Desktop/swati/DSC01415.jpg',
              '/Users/chetandev/Desktop/swati/DSC01634.jpg',
              '/Users/chetandev/Desktop/swati/IMG_0398.jpg',
              '/Users/chetandev/Desktop/swati/dev.jpg',
              ]



# def insert_name(image, name, cursor):
#     draw = ImageDraw.Draw(image, 'RGBA')
#
#     x = cursor[0]
#     y = cursor[1]
#     draw.rectangle([(x, y+200), (x+300, y+240)], (0, 0, 0, 123))
#     draw.text((x+8, y+210), name, (255, 255, 255))

def create_collage(cells, cols=1, rows=1):
    w, h = Image.open(image_info[0]).size
    collage_width = 1000
    collage_height = 1000
    new_image = Image.new('RGB', (collage_width, collage_height))
    cursor = (0,0)
    for cell in cells:
        # place image
        new_image.paste(Image.open(cell), cursor)
        #add name
        # insert_name(new_image, cell['name'], cursor)
        # move cursor
        y = cursor[1]
        x = cursor[0] + w
        if cursor[0] >= (collage_width - w):
            y = cursor[1] + h
            x = 0
        cursor = (x, y)
    new_image.save('/Users/chetandev/Desktop/swati/collage.jpg')

create_collage(image_info, cols=3, rows=3)


# def create_collage(width, height, listofimages):
#     cols = 2
#     rows = 2
#     thumbnail_width = width//cols
#     thumbnail_height = height//rows
#     print thumbnail_height
#     print thumbnail_width
#
#     size = thumbnail_width, thumbnail_height
#     new_im = Image.new('RGB', (width, height))
#     ims = []
#     for p in listofimages:
#         im = Image.open(p)
#         im.thumbnail(size)
#         ims.append(im)
#     i = 0
#     x = 0
#     y = 0
#     for col in range(cols):
#         for row in range(rows):
#             print(i, x, y)
#             new_im.paste(ims[i], (x, y))
#             i += 1
#             y += thumbnail_height
#         x += thumbnail_width
#         y = 0
#
#     new_im.save("/Users/chetandev/Desktop/swati/collage.jpg")
#
# create_collage(430, 300, image_info)

