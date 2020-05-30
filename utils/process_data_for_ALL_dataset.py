import os
from PIL import Image

yourpath = './data/ALL'

# change formats into '.png'
formats = ['.tif', '.jpg', '.tiff']
for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        if os.path.splitext(os.path.join(root, name))[1].lower() in formats:
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".png"):
                # print("A png file already exists for %s" % name)
                None
            # If a png is *NOT* present, create one from other formats.
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".png"
                try:
                    print(os.path.join(root, name))
                    im = Image.open(os.path.join(root, name))
                    print("Generating png for %s" % name)
                    im.thumbnail(im.size)
                    im.save(outfile, "PNG", quality=100)
                    os.remove(os.path.join(root, name))
                except Exception as e:
                    print(e)

# rename av files for HRF dataset
for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        if name.endswith('_AVmanual.png'):
            print(os.path.join(root, name))
            os.rename(os.path.join(root, name),
                        os.path.join(root, name).replace('_AVmanual',''))
