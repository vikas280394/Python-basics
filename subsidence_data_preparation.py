# -*- coding: utf-8 -*-
"""
# Subsidience data preparation code
Created on Tue Apr 21 15:45:01 2020

@author: vikas
"""

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import label

import base64

from osgeo import gdal, osr #gdal later to be used for georeferencing

import sys

 

 

import sklearn

 

import pandas as pd

import numpy as np

import cv2

import os

from PIL import Image,ImageFilter

from PIL import ImageDraw

 

#changing the working directory

 

#Loading an image

## path ::: path where the gif images downloaded from internet are saved

## RGB_Dict_file ::: path where the RGB disctionary is saved

## output ::: path where the results will be saved

 

def gif2tif(path = "V:\VikasSharma\uk_climate\images",RGB_Dict_file = "V:/VikasSharma/uk_climate/new_rgb_dict.xlsx",output = "V:/VikasSharma/uk_climate/result/"   ):

   

    os.chdir(path)

    files =  [f for f in os.listdir(path) if f.endswith('.gif')]

   

    for i in files:

        file_name = i

   

        if os.path.isfile(output + file_name[0:-4] +"labelled" + ".tif"):

            print("File exists.", file_name)

        

        else:

   

        

        

        ################################################################################################################

        ################################################################################################################

        ################################################################################################################

       

        ####################################    Reading the GIF FILE  ################################################################

       

            img = Image.open(file_name) #reading a gif file

            img_convert = img.convert("RGB") #converting a gif  into RGB verison   

            area = (450,155,img_convert.width-20,380)  # Cropping the legend area

            cropped_img = img_convert.crop(area)

            blue_col   = cropped_img.getcolors()

            blue_col.sort(key = lambda tup:tup[0],reverse = True)

            blue_col = [blue_col[0][1]]

           

            

            year = int(file_name[-10:-6])

            maptype = file_name[0:-11]

            maptype = maptype.upper()

            month = int(file_name[-6:-4])

           

        

        

        

            def rgb2hex(r,g,b):

                return "#{:02x}{:02x}{:02x}".format(r,g,b)

       

            #Loading the RGB mapping file into the system

            RGB_mapping = pd.read_excel(RGB_Dict_file)

       

            #Merging the RGB codes to create a single hex value for the color

        

            if maptype == "RAINFALL":

                if maptype == "RAINFALL" and year < 2015:

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="Rainfall_Actual") & (RGB_mapping["YEAR"] < 2015)]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

       

                elif maptype == "RAINFALL" and year == 2015 and month < 6:

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="Rainfall_Actual") & (RGB_mapping["YEAR"] == 2015) & (RGB_mapping["Month#"] <6) ]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

           

                elif maptype == "RAINFALL" and year >= 2015:

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="Rainfall_Actual") & (RGB_mapping["YEAR"] >= 2015) & (RGB_mapping["Month#"] == 6) ]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

               

            elif maptype == "SUNSHINE":

                if maptype == "SUNSHINE" and year < 2015:

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="Sunshine_Actual") & (RGB_mapping["YEAR"] < 2015) & (RGB_mapping["Month#"] == month)]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

       

                elif maptype == "SUNSHINE" and year == 2015 :

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="Sunshine_Actual") & (RGB_mapping["YEAR"] == 2015) & (RGB_mapping["Month#"] == month) ]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

           

                elif maptype == "SUNSHINE" and year > 2015:

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="Sunshine_Actual") & (RGB_mapping["YEAR"] > 2015) & (RGB_mapping["Month#"] == month) ]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

               

            

            

            elif maptype == "MEANTEMP":

                if maptype == "MEANTEMP" and year < 2015:

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="MeanTemp_Actual") & (RGB_mapping["YEAR"] < 2015) & (RGB_mapping["Month#"] == month)]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

       

                elif maptype == "MEANTEMP" and year == 2015 :

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="MeanTemp_Actual") & (RGB_mapping["YEAR"] == 2015) & (RGB_mapping["Month#"] == month) ]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

           

                elif maptype == "MEANTEMP" and year > 2015:

                    RGB_frame = RGB_mapping[(RGB_mapping["TYPE"]=="MeanTemp_Actual") & (RGB_mapping["YEAR"] > 2015) & (RGB_mapping["Month#"] == month) ]

                    area = [ (i,j,i+2,j+2) for i,j in np.asarray(RGB_frame.iloc[:,1:3])]   

        #        color_val = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb_col"] = [ (img_convert.crop((i,j,k,l)).getcolors()[0][1]) for i,j,k,l in area ]

                    RGB_frame["rgb2hex"] = [rgb2hex(col[0],col[1],col[2]) for col in np.asarray(RGB_frame["rgb_col"])]

                   

                    

            else:

                sys.exit("The maptype your are trying to work with is not added in the program and in the RGB Dictionary excel file")

       

        #### Blue Ocean color

            all_col =  blue_col+[i for i in RGB_frame["rgb_col"] ]

            colors_of_interest = all_col[1:10]

       

        

            area_irl = (100,500,200,600)  # Cropping the legend area

            cropped_img_ireland = img_convert.crop(area_irl)

            #cropped_img = img.crop(area)

            #cropped_img_ireland.show()

            ireland_colors = cropped_img_ireland.getcolors()

            ireland_colors.sort(key = lambda tup:tup[0],reverse = True) # sorting the tuple list to get the color which is used the highest

            ireland_colors =  ireland_colors[0][1] # colors or rather areas we are interested in

       

            colo = [rgb2hex(x,y,z) for x,y,z in colors_of_interest ]

            all_colo = [rgb2hex(x,y,z) for x,y,z in all_col ]

            irl_col = rgb2hex(ireland_colors[0],ireland_colors[1],ireland_colors[2]) 

        

            red_knn, green_knn, blue_knn = cv2.split(np.asarray(img_convert))

       

        # Refilling the noise from the actual information

            alpha_channel = np.ones(blue_knn.shape, dtype=blue_knn.dtype) * 50 #creating a dummy alpha channel image.

           

            img_RGBA_knn = cv2.merge((red_knn, green_knn,blue_knn , alpha_channel))

       

            img_RGBA2_knn = Image.fromarray(img_RGBA_knn, 'RGBA')

       

        #in the list we will ignore the first as its the backgrounf blue color

       

        #Converting the image from an RGB type to a color coordinate type where the we have x-coordinate, y- coordinate and the color

            def load_image(filename):

                im = Image.open(filename)

                if im.mode is not 'RGBA':

                    return im.convert(mode='RGBA')

                return im

       

            def convert_to_dataframe(image):

                pixels = image.load()

                data = []

                all_colors = []

                for x in range(0, image.width):

                    for y in range (0, image.height):

                        pixel_color = rgba_to_hex(pixels[image.width - x - 1, image.height - y - 1])

                        data.append([x, y, pixel_color])

                        all_colors.append(pixel_color)

                return data, set(all_colors)

       

            def rgba_to_hex(rgba_tuple):

                assert type(rgba_tuple) == tuple and len(rgba_tuple) == 4

                return "#{:02x}{:02x}{:02x}".format(rgba_tuple[0], rgba_tuple[1], rgba_tuple[2])

       

            data_knn,color_knn =   convert_to_dataframe(img_RGBA2_knn)

       

        #Full data extracted from the image with xy coordinate and the color of the coordinate

            df_knn = pd.DataFrame(data_knn, columns=['x', 'y', 'color'])

       

        #'#C8C8C8'

        #colo_hexa =  ['#663300','#ccccff','#996633','#3366ff','#ffffff','#cc9966','#9999ff','#330000','#000099','#bddcbd' ]

            colo.append(irl_col)

            colo_hexa =  colo

            all_colo.append(irl_col)

        #color_hexa2 = ['#99ccff','#663300','#ccccff','#996633','#3366ff','#ffffff','#cc9966','#9999ff','#330000','#000099','#bddcbd' ]

            color_hexa2 = all_colo

       

        #Building the training dataset consisting of color of our legends             

            train_knn =   df_knn[ df_knn.color.isin(colo_hexa)]

            train_knn = pd.DataFrame(train_knn)

       

        #Ocean data consisiting of only the pixels which constitute ocean

            ocean_data = df_knn[df_knn.color == all_colo[0] ]

            ocean_data = pd.DataFrame(ocean_data)

                           

        #test_knn = df_knn[~df_knn.color.isin(color_hexa2)]

           

            

            if file_name[0:8] == "sunshine":

        #Test dataset- dataset which is mdae up of noise and needs to be filled with the actual legend colors

                test_knn = df_knn

                test_knn = pd.DataFrame(test_knn)

                clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=9)

            else:

                test_knn = df_knn[~df_knn.color.isin(color_hexa2)]

                test_knn = pd.DataFrame(test_knn)

                clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4)

               

            clf.fit(train_knn[['x', 'y']], train_knn['color'])

       

        #predicting the test data

            preds = clf.predict(test_knn[['x', 'y']])

            test_knn.color = preds

       

        #data2 = train.append(test)

       

            size =  (img_RGBA2_knn.width, img_RGBA2_knn.height)

       

        #def save_to_file(filename, dataframes, size):

        #    ni = Image.new('RGBA', size, '#000000')

        #    pixels = ni.load()

        #    for df in dataframes:

        #        for row in df.itertuples():

        #            pixels[int(size[0] - 1 - row.x), int(size[1] - 1 - row.y)] = hex_to_rgba(row.color)

        #    ni.save(filename)

       

        # Converting the image format from Hex to RGB format

            def hex_to_rgba(hex_string):

                return int(hex_string[1:3], 16), int(hex_string[3:5], 16), int(hex_string[5:7], 16), 255

       

        # Building the image after combining the train, test and ocean data

            ni = Image.new('RGBA', size, '#000000')

            pixels = ni.load()

            for df in [train_knn,test_knn,ocean_data]:

                for row in df.itertuples():

                    pixels[int(size[0] - 1 - row.x), int(size[1] - 1 - row.y)] = hex_to_rgba(row.color)

       

        #save_to_file('{}_combined{}'.format('file', '.png'), [train_knn,test_knn,ocean_data], (img_RGBA2_knn.width, img_RGBA2_knn.height))

       

        #ni.save("experiment.png")

        # Removing the unwated pictures from the Image like name title etc           

            red, green, blue, a = cv2.split(np.asarray(ni)) #Splitting the value of RGB

       

        

            a_red = [x for x,y,z in colors_of_interest]

            a_red.append(1) #adding the pixels of black color for later stages

       

            a_green = [y for x,y,z in colors_of_interest]

            a_green.append(1)

       

            a_blue = [z for x,y,z in colors_of_interest]

            a_blue.append(2)

       

        

            a_red_unt8 = np.array(a_red)

            a_red_unt8 = a_red_unt8.astype(np.uint8)

            a_green_unt8 = np.array(a_green)

            a_green_unt8 = a_green_unt8.astype(np.uint8)

            a_blue_unt8 = np.array(a_blue)

            a_blue_unt8 = a_blue_unt8.astype(np.uint8)

                

        #ID_color = [sum(x) for x in zip(a_red,2*np.array(a_green),np.array(a_blue)*3)]      

            ID_color = a_red_unt8 + 2*(a_green_unt8) + 3*(a_blue_unt8)

       

            ID_array = red+2*(green)+3*(blue)

       

        #red_new =  np.where( np.isin(red,a_red) == True,red,0)

        #green_new = np.where( np.isin(green,a_green) == True,green,0)

        #blue_new = np.where( np.isin(blue,a_blue) == True,blue,0)

       

            red_new =  np.where( np.isin(ID_array,ID_color) == True,red,86)

            green_new = np.where( np.isin(ID_array,ID_color) == True,green,0)

            blue_new = np.where( np.isin(ID_array,ID_color) == True,blue,11)

       

        

            new_image = cv2.merge((red_new,green_new,blue_new))   

        #cv2.imshow("image",bi)

        #cv2.waitKey(0)

        #cv2.destroyAllWindows()  

        

            img = Image.fromarray(new_image, 'RGB')

       

            img.paste((86,0,11),[0,0,320,90]) # removing the title

            img.paste((86,0,11),[0,0,img.width,15]) # removing the white noise at the top

            img.paste((86,0,11),[0,0,15,img.height]) # removing the white noise at the left

            img.paste((86,0,11),[0,760,img.width,770]) # removing the white noise at the bottom

            img.paste((86,0,11),[600,0,img.width,770]) # removing the white noise at the right

            img.paste((86,0,11),[0,650,150,770]) # removing the copyright

            img.paste((86,0,11),[440,0,img.width,390]) # removing the legend

       

        

        # Saving the PNG File

            img.save(file_name[0:-4]+".png")

        #################################################################################################

        ##############################    GEOREFERENCING         ########################################

        #################################################################################################

       

            

            dst_filename = output + file_name[0:-4] + ".tif"

           

        

        

        # Opens source dataset

            src_ds = gdal.Open(file_name[0:-4]+".png")

       

            format = "GTiff"

            driver = gdal.GetDriverByName(format)

        # Open destination dataset

            dst_ds = driver.CreateCopy(dst_filename, src_ds, 0)

       

        

        # Specify raster location through geotransform array

        # (uperleftx, scalex, skewx, uperlefty, skewy, scaley)

        # Scale = size of one pixel in units of raster projection

        # this example below assumes 100x100

       

        #For year before 2015

            if int(file_name[-10:-6]) >= 2015:

                    gt = [-223513.20509390928782523,1517.06391576899545726,0,1117758.4120618924498558,0,-1513.13848085692870882]

            else:

                gt = [-222938.72198931037564762,1509.12023242977852533,0,1112929.1783957164734602,0,-1500.89328521621064283]

       

        #for year starting 2015

        # Set location

            dst_ds.SetGeoTransform(gt)

       

        # Get raster projection

            epsg = 27700

            srs = osr.SpatialReference()

            srs.ImportFromEPSG(epsg)

            dest_wkt = srs.ExportToWkt()

       

        # Set projection

            dst_ds.SetProjection(dest_wkt)

       

        # Close files

            dst_ds = None

            src_ds = None

            os.remove(file_name[0:-4]+".png")

       

        

        

        

        #################################################################################################

        ##############################    RELABELING         ########################################

        #################################################################################################

       

        

        

           # RGB_mapping_hex = [ (a,rgb2hex(b,c,d),e,h)  for a,b,c,d,e,f,g,h,i,j in np.asarray(RGB_mapping)]

       

        

            gdal.UseExceptions()

            driver = gdal.GetDriverByName('GTiff')

            file = gdal.Open(dst_filename)

            band1 = file.GetRasterBand(1)

            band2 = file.GetRasterBand(2)

            band3 = file.GetRasterBand(3)

        # Extracting the array of dirrent colors from the tif file , mainly the RGB colors

            listr = band1.ReadAsArray()

            listrn = listr.astype(np.int32)

            listg = band2.ReadAsArray()

            listgn = listg.astype(np.int32)

            listb = band3.ReadAsArray()

            listbn = listb.astype(np.int32)

            new_array = band1.ReadAsArray()

       

        #    if maptype == "rainfall":

        #        col_month = [(j,k) for i,j,k,l in RGB_mapping_hex if i == "rainfall" ]

        #    else:

        #        col_month = [(j,k) for i,j,k,l in RGB_mapping_hex if i == maptype and int(l) == int(month) ]

                   

        # reclassification

        # populating our legend colors with their actual legend values

            for j in  range(file.RasterXSize):

                for i in  range(file.RasterYSize):

                    if rgb2hex(listrn[i,j],listg[i,j],listb[i,j]) in [ col_hex for col_hex in np.asarray(RGB_frame["rgb2hex"])]:

                        listrn[i,j] = [ val2 for col2,val2 in np.asarray(RGB_frame[["rgb2hex","LEGEND"]]) if col2 == rgb2hex(listrn[i,j],listg[i,j],listb[i,j])] [0]

                        listgn[i,j] = -999

                        listbn[i,j] = -999

                    else:

                        listrn[i,j] = -999

                        listgn[i,j] = -999

                        listbn[i,j] = -999

                           

        # Saving it to a new raster file

            new_raster_name = output + file_name[0:-4] +"labelled" + ".tif"

            file2 = driver.Create( new_raster_name, xsize=file.RasterXSize , ysize=file.RasterYSize , bands=1, eType=gdal.GDT_Int16)

       

            file2.GetRasterBand(1).WriteArray(listrn)

       

        # spatial ref system

        # Saving the geo referencing of the image to old file to this file.   

            proj = file.GetProjection()

            georef = file.GetGeoTransform()

            file2.SetProjection(proj)

            file2.SetGeoTransform(georef)

            file2.FlushCache()

            file2 = None

            file = None

            os.remove(dst_filename)

   

    

    

    

    #########################################

        ###########################################

        ###########################################

        ###########################################

   

       

gif2tif(path = "V:\VikasSharma\uk_climate\images",RGB_Dict_file = "V:/VikasSharma/uk_climate/new_rgb_dict.xlsx",output = "V:/VikasSharma/uk_climate/result/"   )

 