import matplotlib as mlp
import numpy as np
import pandas as pd
import flirimageextractor
import os
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image


def get_date_taken(path):
  ''' Get the date that the image was taken if it is in the Metadata'''
  try:
    date = Image.open(path)._getexif()[36867]
    return date
  except:
    return None

def build_duo_image_links(img_dir):
  '''Extract DateTimes for each image in directory'''
  imgs = os.listdir(img_dir)    ## List all images in directory
  df = pd.DataFrame(imgs)       ## Build dataframe with names of all images

  # Loop over all images to get date and time.
  for i, row in df.iterrows():  
    date = pd.to_datetime(get_date_taken(os.path.join(img_dir, row[0])), 
                          format='%Y:%m:%d %H:%M:%S')
    df.loc[df.index == i, 'DateTime'] = date

  df.sort_values(by='DateTime', inplace=True) ## Let's sort the dataframe
  df.reset_index(drop=True, inplace=True)     ## Now drop the old index

  return df

def get_flir_image(imgPath):
  ''' Will return either the flir or rgb image '''
  try:
    flir = flirimageextractor.FlirImageExtractor(palettes=[mlp.cm.jet, mlp.cm.bwr, mlp.cm.gist_ncar])
    flir.process_image(imgPath)
    flir.save_images()
    data = flir.get_metadata(imgPath)
    img = flir.thermal_image_np
    modality = 'thermal'
  except:
    img = mlp.pyplot.imread(imgPath)
    modality = 'rgb'
  
  return img, modality

def plot_rgb_thermal(thermal, rgb):
  fig = make_subplots(rows=1, cols=2)

  fig.add_trace(go.Image(z=rgb), 1, 1)
  fig.add_trace(go.Heatmap(z=thermal), 1, 2)

  fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)

  return fig
