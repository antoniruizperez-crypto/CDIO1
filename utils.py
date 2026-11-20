import rasterio
import matplotlib.pyplot as plt
import sys
import numpy as np

def show_geotiff(tif_path, downsample_normalize=True, title=""):
    with rasterio.open(tif_path) as src:
        img = src.read(1)
        transform = src.transform

        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap='gray')
        plt.colorbar(im)
        plt.title(title)

        clicked_coords = []

        def onclick(event):
            if event.inaxes != ax:
                return
            x_pix, y_pix = int(event.xdata), int(event.ydata)
            val = img[y_pix, x_pix]

            # Average value in the 8 pixels around it
            avg_val = np.mean(img[max(0, y_pix-1):y_pix+2, max(0, x_pix-1):x_pix+2])

            lon, lat = rasterio.transform.xy(transform, y_pix, x_pix)
            print(f"Clicked pixel: ({x_pix}, {y_pix})")
            print(f"Coordinates: ({lon:.6f}, {lat:.6f})")
            print(f"Pixel Value: {val}")
            print(f"Average Value (3x3 window): {avg_val:.2f}")
            clicked_coords.append((lon, lat, val))
            plt.close(fig)  # close after one click

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()  # waits for the user to click

        if clicked_coords:
            return clicked_coords[0]
        else:
            return None



if __name__ == "__main__":
    """ 
        Usage: python utils.py <path_to_geotiff>. 
        E.g python utils.py S2B_MSIL2A_20250107T051119_N0511_R019_T43QHU_20250107T080132_green.tif
    """
    show_geotiff(sys.argv[1])
