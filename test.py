import buzzard as buzz
import numpy as np
import matplotlib.pyplot as plt


rgb_path = 'austin1.tif'

ds = buzz.DataSource(allow_interpolation=True)
ds.open_raster('rgb', rgb_path)

# footprint is the information of the image :
#top left coordinates, bottom right coordinates, size (height and wdith)
# and rsize (heighth and width) : size of the pixel image
fp = buzz.Footprint(
    tl=ds.rgb.fp.tl,
    size=ds.rgb.fp.size,
    rsize=ds.rgb.fp.rsize,
)


tl=ds.rgb.fp.tl

print(ds.rgb.fp.gt)

with buzz.Env(warnings=0, allow_complex_footprint=1):
    fp = buzz.Footprint(gt=(tl[0], 0.1, 0, tl[1], 0, -.1),rsize=ds.rgb.fp.rsize)



    # band is the order of canal (red green blue)
    rgb = ds.rgb.get_data(band=(1, 2, 3), fp=fp).astype('uint8')
    print(rgb.shape)
    # Show image with matplotlib and descartes
    fig = plt.figure()
    plt.title('Test raster')
    ax = fig.add_subplot(111)
    ax.imshow(rgb, extent=[fp.lx, fp.rx, fp.by, fp.ty])
    plt.show()