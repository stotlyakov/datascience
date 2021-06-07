import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
from app import app
import base64

cwd = os.getcwd()

# Define display function to show scatter plot of the color location segments
def scater_color_segments(vectorized, label, center):
    colors = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']
    plt.figure(2)
    plt.figure(figsize=(15,15))
    for x in range(len(center)):
        A = vectorized[label.ravel()==x]
        plt.scatter(A[:,0], A[:,1], c = colors[x])
    
    plt.scatter(center[:,0],center[:,1],s = 100,c = 'y', marker = '+')
    plt.xlabel('X - color range 0-255'),plt.ylabel('Y - color range 0-255')
    
    plt.savefig(cwd + "/assets/scatterK3.png", dpi=96)

def show_pca_compressed(components):
    pca = PCA(components)
 
    #Applying to red channel and then applying inverse transform to transformed array.
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)

    #Applying to Green channel and then applying inverse transform to transformed array.
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)

    #Applying to Blue channel and then applying inverse transform to transformed array.
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)
    
    #When compressing the Image inverse Transformation is necessary to recreate the original dimensions of the base image.
    #In the process of reconstructing the original dimensions from the reduced dimensions, some information is lost as 
    #we keep only selected principal components, 20, 100, 400 in this case.
    
    #Stacking the inverted arrays using dstack function. Here it is important to specify the datatype of
    #our arrays, as most images are of 8 bit. Each pixel is represented by one 8-bit byte.
    img_compressed = (np.dstack((red_inverted, green_inverted, blue_inverted))).astype(np.uint8)
    return img_compressed

original_image = cv2.imread(cwd + "/assets/image.jpeg")
figOriginalPic = go.Figure(data=[go.Image(z=original_image)])

figOriginalPic.update_layout(title_text='Our original image in RGB color space')

img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
figImgHsv = go.Figure(data=[go.Image(z=img)])
figImgHsv.update_layout(title_text='Image converted to HSV')

vectorizedOriginal = img.reshape((-1,3))
vectorized = np.float32(vectorizedOriginal)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


#3 clusters
K = 3
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
scater_color_segments(vectorized, label, center)

#reshape the image so we can display it
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))

plt.figure(3)
plt.figure(figsize=(1020/96,300/96))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.savefig(cwd + "/assets/original_segmented_K3.jpeg",dpi=96)

#7 clusters
K = 7
attempts=10
ret7,label7,center7=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center7 = np.uint8(center7)
res7 = center7[label7.flatten()]
result_image7 = res7.reshape((img.shape))

plt.figure(4)
plt.figure(figsize=(1020/96,300/96))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image7)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.savefig(cwd + "/assets/original_segmented_K7.jpeg",dpi=96)


#20 clusters
K = 20
attempts=10
ret20,label20,center20=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center20 = np.uint8(center20)
res20 = center20[label20.flatten()]
result_image20 = res20.reshape((img.shape))

plt.figure(5)
plt.figure(figsize=(1020/96,300/96))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image20)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.savefig(cwd + "/assets/original_segmented_K20.jpeg",dpi=96)


# Define display function to show 3D scatter plot of the color location segments RGB(255,255,255)
def scater_color_segments3D(vectorized, label, center):
    
    for x in range(len(center)):
        A = vectorized[label.ravel()==x]
        B = vectorizedOriginal[label.ravel()==x]
        mydata = go.Scatter3d(x = A[:,0], y = A[:,1], z = A[:,2], 
                      mode='markers', 
                      marker=dict(
                            size=2,
                            color= B[:,2],          # set color to an array/list of desired values
                            colorscale='Viridis',   # choose a colorscale
                            opacity=0.8
                        )
                     )
    return go.Figure([mydata])

#scater_color_segments(vectorized, label, center)
k3Scater3d = scater_color_segments3D(vectorized, label, center)

# Splitting the image in R,G,B arrays.
blue,green,red = cv2.split(original_image) 
#it will split the original image into Blue, Green and Red arrays.

imgeCompressed_20 = go.Figure(data=[go.Image(z=show_pca_compressed(20))], layout_title_text = "PCA with 20 components")
imgeCompressed_100 = go.Figure(data=[go.Image(z=show_pca_compressed(100))], layout_title_text = "PCA with 100 components")
imgeCompressed_400 = go.Figure(data=[go.Image(z=show_pca_compressed(400))], layout_title_text = "PCA with 400 components")
imgeCompressed_675 = go.Figure(data=[go.Image(z=show_pca_compressed(675))], layout_title_text = "PCA with 675 (Max) components")

layout = html.Div([dbc.Container([
    
        dbc.Row([dbc.Col(dbc.Card(
        dbc.Row([
        dcc.Link(html.A('GitHub'), href="https://github.com/stotlyakov/datascience/blob/main/apps/hw8.py", style={'color': 'white', 'text-decoration': 'underline'}, target="_blank"),
        dcc.Link(html.A('Testing data set: int_rates_testing.csv'), href="https://github.com/stotlyakov/datascience/blob/main/data/int_rates_testing.csv", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank"),
        dcc.Link(html.A('Training data set: int_rates_training.csv'), href="https://github.com/stotlyakov/datascience/blob/main/data/int_rates_training.csv", style={'color': 'white', 'text-decoration': 'underline', 'margin-left':'10px','margin-right':'10px'},target="_blank")
        ]),

        body=True, color="dark"))]),
        html.Br(),
        dbc.Row([dbc.Col(html.H4(children='Original image 1200x675 px'), className="mb-2")]),
        html.Br(),
        html.Img(src="/assets/image.jpeg", height='300px'),
        html.Br(),
        html.Br(),
        dbc.Row([dbc.Col(html.H4(children='Load the original image with opencv-python'), className="mb-2")]),
        html.Pre(children = [html.Label(children = 'PY'), 
                             html.Code(children= "cwd = os.getcwd()\n" + 
                             "noriginal_image = cv2.imread(cwd + '/assets/image.jpeg')\n" +
                             "figOriginalPic = go.Figure(data=[go.Image(z=original_image)\n\n" +
                             "#convert to HSV\n" +
                             "img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)])\n"
                             )], className= 'code code-py'),

        dbc.Row([dbc.Col(html.H6(children='We need to convert our image from RGB Colours Space to HSV becasue R, G, and B components of an object’s color hue/lightness/chroma ' +
            'or hue/lightness/saturation are often more relevant to describe the image.'), className="mb-4")]),

        dcc.Graph(id="graphfigOriginalPic", figure = figOriginalPic, style={ "width" :"530px", "margin-bottom":"10px","display": "inline-block"}),
        dcc.Graph(id="graphfigImgHsv", figure = figImgHsv, style={ "width" :"530px", "margin-bottom":"10px","margin-left":"30px","display": "inline-block"}),

        html.Br(),
        dbc.Row([dbc.Col(html.H4(children='Turn each RGB chanel into vecor so we can apply clustering. We need to have the RGB valuse as floats'), className="mb-2")]),
        html.Pre(children = [html.Label(children = 'PY'), 
                             html.Code(children= "vectorizedOriginal = img.reshape((-1,3))\n" + 
                             "vectorized = np.float32(vectorizedOriginal)\n" +
                             "criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)\n"
                             )], className= 'code code-py'),
        html.Br(),
        dbc.Row([dbc.Col(html.H4(children='Use cv2.kmeans, ref here: https://docs.opencv.org/master/d5/d38/group__core__cluster.html'), className="mb-4")]),
        dbc.Row([dbc.Col(html.H4(children='Create 3 clusters and do 10 iterations.'), className="mb-4")]),
        dbc.Row([dbc.Col(html.H6(children='ret : It is the sum of squared distance from each point to their corresponding centers.'), className="mb-4")]),
        dbc.Row([dbc.Col(html.H6(children='label : This is the label array where each element marked "0","1",.....'), className="mb-4")]),
        dbc.Row([dbc.Col(html.H6(children='center : This is array of centers of clusters.'), className="mb-4")]),

        html.Br(),
        html.Pre(children = [html.Label(children = 'PY'), 
                            html.Code(children= "K = 3\n" + 
                            "attempts=10\n" +
                            "ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)\n"
                            )], className= 'code code-py'),


        dbc.Row([dbc.Col(html.Img(src="/assets/scatterK3.png", width='500px')), dbc.Col(dcc.Graph(id="graphfigOriginalPic", figure = k3Scater3d, style={ "width" :"500px", "height": "500px"}))], style={"padding":"25px"}),
        dbc.Row(dbc.Col(html.Img(src="/assets/original_segmented_K3.jpeg")), style={"padding":"30px"}),


        dbc.Row([dbc.Col(html.H4(children='Use 7 clusters'), className="mb-2")]),
        dbc.Row(dbc.Col(html.Img(src="/assets/original_segmented_K7.jpeg")), style={"padding":"30px"}),

        dbc.Row([dbc.Col(html.H4(children='Use 20 clusters'), className="mb-2")]),
        dbc.Row(dbc.Col(html.Img(src="/assets/original_segmented_K20.jpeg")), style={"padding":"30px"}),

        dbc.Row([dbc.Col(html.H6(children='What we oberve is that with the increase in the value of K, the image becomes clearer because the K-means algorithm can classify more cluster of colors.'), className="mb-4")]),
        html.Br(),

        dbc.Row([dbc.Col(dbc.Card(html.H3(children='Now lets compress the image using PCA and cv2', className="text-center text-light bg-dark"), body=True, color="dark"))]),
        html.Br(),
    

        html.Br(),
        dbc.Row([dbc.Col(html.H4(children='Apply Principal Components to Individual Arrays.'), className="mb-2")]),
        html.Pre(children = [html.Label(children = 'PY'), 
                             html.Code(children= 
                                "# Splitting the image in R,G,B arrays.\n" +
                                "blue,green,red = cv2.split(original_image) \n\n" +
                                "def show_pca_compressed(components):\n" +
                                    "   pca = PCA(components)\n\n" +
 
                                    "   #Applying to red channel and then applying inverse transform to transformed array.\n" +
                                    "   red_transformed = pca.fit_transform(red)\n" +
                                    "   red_inverted = pca.inverse_transform(red_transformed)\n\n" +

                                    "   #Applying to Green channel and then applying inverse transform to transformed array.\n" +
                                    "   green_transformed = pca.fit_transform(green)\n" +
                                    "   green_inverted = pca.inverse_transform(green_transformed)\n\n" +

                                    "   #Applying to Blue channel and then applying inverse transform to transformed array.\n" +
                                    "   blue_transformed = pca.fit_transform(blue)\n" +
                                    "   blue_inverted = pca.inverse_transform(blue_transformed)\n\n" +
    
                                    "   #When compressing the Image inverse Transformation is necessary to recreate the original dimensions of the base image.\n" +
                                    "   #In the process of reconstructing the original dimensions from the reduced dimensions, some information is lost as \n" +
                                    "   #we keep only selected principal components, 20, 100, 400 in this case.\n\n" +
    
                                    "   #Stacking the inverted arrays using dstack function. Here it is important to specify the datatype of\n" +
                                    "   #our arrays, as most images are of 8 bit. Each pixel is represented by one 8-bit byte.\n" +
                                    "   img_compressed = (np.dstack((red_inverted, green_inverted, blue_inverted))).astype(np.uint8)\n" +
                                    "   plt.imshow(img_compressed)\n\n"

                                    "imgeCompressed_20 = go.Figure(data=[go.Image(z=show_pca_compressed(20))], layout_title_text = 'PCA with 20 components')\n" +
                                    "imgeCompressed_100 = go.Figure(data=[go.Image(z=show_pca_compressed(100))], layout_title_text = 'PCA with 100 components')\n" +
                                    "imgeCompressed_400 = go.Figure(data=[go.Image(z=show_pca_compressed(400))], layout_title_text = 'PCA with 400 components')\n" +
                                    "imgeCompressed_675 = go.Figure(data=[go.Image(z=show_pca_compressed(675))], layout_title_text = 'PCA with 675 (Max) components')"
                             )], className= 'code code-py'),
        html.Br(),
        dbc.Row([dbc.Col(html.H4(children='As we can see the higher the number of components, the higher the qulity of the picture.'), className="mb-2")]),
        html.Br(),

        dcc.Graph(id="graphimgeCompressed_20", figure = imgeCompressed_20, style={ "width" :"530px", "margin-bottom":"10px","display": "inline-block"}),
        dcc.Graph(id="graphimgeCompressed_100", figure = imgeCompressed_100, style={ "width" :"530px", "margin-bottom":"10px","display": "inline-block"}),
        dcc.Graph(id="graphimgeCompressed_400", figure = imgeCompressed_400, style={ "width" :"530px", "margin-bottom":"10px","display": "inline-block"}),
        dcc.Graph(id="graphimgeCompressed_675", figure = imgeCompressed_675, style={ "width" :"530px", "margin-bottom":"10px","display": "inline-block"}),

        html.Br(),
         ])])
