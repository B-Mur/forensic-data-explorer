import matplotlib as mlp
import numpy as np
import pandas as pd
import flirimageextractor
import os
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image
import datetime


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
  img = [img for img in imgs if img.endswith('.jpg')]
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

def plot_rgb_thermal(thermal, rgb, dynamic=None):
  fig = make_subplots(rows=1, cols=2)

  fig.add_trace(go.Image(z=rgb), 1, 1)
  if not dynamic is None: 
    fig.add_trace(go.Heatmap(z=thermal,  zmin=dynamic[0], zmax=dynamic[1], ), 1, 2)
  else:
    fig.add_trace(go.Heatmap(z=thermal), 1, 2)

  fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)

  return fig


def plot_logger_fig(imgDir, loggerFile, thermal_image=None):
  ''' This will return a plot of the logger info with the center value of a thermal image (if provided '''
  logger_df = pd.read_csv(os.path.join(imgDir, loggerFile[0]))
  vals = [datetime.datetime(row.year, row.month, row.day, row.hour, row.minute, row.second) for index, row in logger_df.iterrows()]
  logger_df['FormatDate'] = vals
  logger_fig = go.Figure()
  logger_fig.add_trace(go.Scatter(x=logger_df['FormatDate'], y=logger_df['Surface abdomen'], name='Surface Abdomen'))
  logger_fig.add_trace(go.Scatter(x=logger_df['FormatDate'], y=logger_df['Surface thigh'], name='Surface thigh'))
  if not thermal_image is None:
    logger_fig.add_trace(go.Scatter(x=[date], y=[thermal[90, 120]], name='Current Sample'))
  return logger_fig
