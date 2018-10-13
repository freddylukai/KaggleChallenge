from os import listdir
from os.path import isfile, join
import pydicom
from PIL import Image
import os


path = '../../input/stage_1_train_images'
jpgpath = join(path, 'jpgs')

for idx, f in enumerate(listdir(path)):
    filepath = join(path, f)
    if isfile(filepath):
        aux = pydicom.read_file(filepath)
        im = Image.fromarray(aux.pixel_array)
        name = os.path.basename(filepath)
        name, _ = os.path.splitext(name)
        name = "%s.jpeg" % name
        im.save(join(jpgpath, name))

    if idx > 0 and idx % 100 == 0:
        print(idx)

