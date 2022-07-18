'''app.py

Run this app with `python app.py` and visit http://127.0.0.1:8050/ in your web 
browser.
'''

import argparse
import glob
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import mpl_toolkits.basemap as basemap

from dash_components import navbar
from helper import readAIDA

app = Dash(
    __name__,
    # external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/css/bootstrap.min.css"]
    # assets_external_path = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.2/dist/css/bootstrap.min.css",
    title = "AIDA Dashboard")

def plotOptions(files, vmax=None, mean=False, what='tec',
                tecu=True, group=False, difference=False, res=5, log=False,
                neqDiff=False):
    ''' Routines for selecting plot options

    Mostly follows helpers.plotAnalysis, but without calling helpers.gridPlot
    
    Parameters
    ==========
    filenames : array of strings
        AIDA HDF file paths

    Arguments
    =========
    vmax : float
        The maxmimum value to use in the colorbar when plotting
        (Default: None)
    mean : Boolean
        Flag to set whether the AIDA HDF file is mean collapsed or not
        (Defaut: False)
    what : ['nmf2','hmf2','tec']
        Set what to plot
        (Default: 'tec')
    tecu : Boolean
        Set whether to plot in TECU or not
        (Default: True)
    difference : Boolean
        If true, plot difference between analysis and background
        (Default: False)
    res : float
        What resolution to plot the output at (in degrees)
        (Default: 5)
    log : Boolean
        Set if you want to plot in log space
    neqDiff : Boolean
        Set if you want to plot the difference between the analysis and
        standard NeQuick

    Example
    =========
    TODO: insert example
    TODO: what is the group parameter?
    '''
    
    # import runAIDA as rt # TODO: current environment does not support cartopy, so oNeQuick cannot run and neqDiff option not available
    # Assume that the lon,lat and alt is the same for all files
    lon,lat,alt,time,f107 = readAIDA(files[0], ['lon','lat','alt','datetime',
                                    'f107'], group=False, mean=False)  

    inputs = dict(lons=lon,
                  lats=lat,
                  alts=alt,
                  debug=False)
    # t0 = dt.datetime.strptime(np.str(time, 'utf-8'), '%Y%m%d%H%M%S')

    if log == True and tecu == True:
        # We know best, they don't *really* want this in TECU:
        tecu = False
    
    for n,fn in enumerate(files):
        # logger.debug('Making plot %s of %s', n+1, len(filenames))
        print('Making plot {} of {}'.format(n+1, len(files)))
        time,f107 = readAIDA(fn, ['datetime','f107'],
                                group=False, mean=False)
        # t0 = dt.datetime.strptime(np.str(time, 'utf-8'), '%Y%m%d%H%M%S')

        if log:
            nea = np.log10(readAIDA(fn, 'analysis', group=group, mean=mean))
            neb = np.log10(readAIDA(fn, 'background', group=group, mean=mean))
        else:
            nea = readAIDA(fn, 'analysis', group=group, mean=mean)
            neb = readAIDA(fn, 'background', group=group, mean=mean)

        # if neqDiff:
            # neDef = rt.runONeQ(inputs, t0=t0, f107=f107)
            # gridPlot(nea-neDef,lon,lat,alt,what,res=res,vmax=vmax,tecu=tecu,
            #          save=os.path.join(saveLoc,np.str(n).zfill(4)+'.png'))
        # elif difference:
        if difference:
            den = nea - neb
            # gridPlot(nea-neb,lon,lat,alt,what,res=res,vmax=vmax,tecu=tecu,
            #          save=os.path.join(saveLoc,np.str(n).zfill(4)+'.png'))
        else:
            if mean:
                den = nea
                # print("plotOptions: den.max()={}".format(den.max()))
                # gridPlot(nea,lon,lat,alt,what,res=res,vmax=vmax,tecu=tecu,
                #          save=os.path.join(saveLoc,np.str(n).zfill(4)+'.png'))
            else:
                den = 10**(np.mean(np.log10(nea),axis=-1))
                # gridPlot(10**(np.mean(np.log10(nea),axis=-1)),lon,lat,alt,what,res=5,
                #          save=os.path.join(saveLoc,np.str(n).zfill(4)+'.png'),vmax=vmax,
                #          tecu=tecu)
        print("plotOptions complete")
        return den, nea, neb,  lon, lat, alt, what, vmax, tecu

def plotPreprocess(den, nea, neb, lon, lat, alt, what, res=None, title=None, units=None,
             time=None, vmin=None, vmax=None, save=None, tecu=False,
             lonMin=None, lonMax=None, latMin=None, latMax=None):
    '''Routines for preprocessing AIDA outputs, ready for plotting

    Mostly follows the preprocessing routines of helpers.gridPlot without the plotting part

    Parameters
    ==========
    den : array of floats
        A 3D array of densities (lon, lat, alt)
    lon : array of floats
        1D array of longitude values
    lat : array of floats
        1D array of latitude values
    alt : array of floats
        1 or 3D array of altitude values (galt from dengrid object)
    what : string
        What you want to plot (NmF2, hmF2, max, TEC)
    res : float, optional
        Resolution of the output map (interpolated if necessary). If None
        use built in resolution of the file
        (Default: None)
    time : datetime object, optional
        If provided add the solar terminator
        (Default: None)
    title : string, optional
        The title to use on the plot
        (Default: None)
    units : string, optional
        The units to use for the colourbar
        (Default: None)
    vmin : float, optional
        The lowest value for the colourbar. If None it is calculated
        automatically
        (Default: None)
    vmax : float, optional
        The largest value for the colourbar. If None it is calculated
        automatically
        (Default: None)
    save : string, optional
        If this is not 'None' then figure is saved to that location
        (Default: None)
    tecu : boolean, optional
        If set then the calculated integrated value (by using what='tec') is
        divided by 1e16 to get a value in TECU
        (Default: False)
    lonMin : float, optional
        If passed (along with lonMax,latMin and latMax) rather than drawing
        a global map a regional map is made. This is the minimum value of
        longitude in the regional map.
        (Default: None)
    lonMax : float, optional
        If passed (along with lonMin,latMin and latMax) rather than drawing
        a global map a regional map is made. This is the maximum value of
        longitude in the regional map.
        (Default: None)
    latMin : float, optional
        If passed (along with lonMin,lonMax and latMax) rather than drawing
        a global map a regional map is made. This is the minimum value of
        latitude in the regional map.
        (Default: None)
    latMax : float, optional
        If passed (along with lonMin,lonMax and latMin) rather than drawing
        a global map a regional map is made. This is the maximum value of
        latitude in the regional map.
        (Default: None)

    '''

    alt = np.array(alt, ndmin=1)
    if alt.ndim == 1:
        # logger.debug('Altitude dimension is only 1, so use meshgrid to expand')
        print('Altitude dimension is only 1, so use meshgrid to expand')
        glon, glat, galt = np.meshgrid(lon, lat, alt, indexing='ij')
    else:
        galt = alt

    # Check if all regional map points have been passed (otherwise plot global)
    if (lonMin is None) or (lonMax is None) or (latMin is None) or (
            latMax is None):
        globMap = True
    else:
        globMap = False

    if what.lower() == 'nmf2' or what.lower() == 'max':
        # logger.debug('Requested a max value')
        print('Requested a max value')
        lonlat = np.max(den, axis=2)
        if units:
            label_name = units
        else:
            label_name = ''
    elif what.lower() == 'hmf2' or what.lower() == 'heightmax':
        # logger.debug('Requested height of max value')
        print('Requested height of max value')
        ind2 = np.argmax(den, axis=2)
        ind0, ind1 = np.meshgrid(np.arange(len(lon)),np.arange(len(lat)),
                                 indexing='ij')
        lonlat = galt[ind0, ind1, ind2]
        label_name = 'Height of maximum value (km)'
    elif what.lower() == 'tec':
        # logger.debug('Requested a TEC map')
        print('Requested a TEC map')
        alt_diff = np.concatenate((np.zeros((galt.shape[0], galt.shape[1], 1)),
                                   np.cumsum(np.diff(galt, axis=2), axis=2)),
                                   axis=2)
        # altitude is in metres
        lonlat = np.trapz(den, alt_diff*1000, axis=2)
        if units:
            label_name = units
        else:
            if tecu:
                lonlat = lonlat/1e16
                label_name = 'Total Electron Content (TECU)'
            else:
                label_name = 'Total Electron Content'
    
    # Basemap requires grid to be lat x lon
    latlon = lonlat.swapaxes(0, 1)

    # Check required resolution
    y_array = lon
    x_array = np.arange(len(y_array))
    dy = np.zeros(y_array.shape, float)
    dy[0:-1] = np.diff(y_array) / np.diff(x_array)
    # Need to add on the last value
    dy[-1] = (y_array[-1] - y_array[-2])/(x_array[-1] - x_array[-2])
    existingRes = dy.max()

    if res == None:
        lons, lats = np.meshgrid(lon,lat)
    elif res != existingRes:
        # logger.debug('Interpolating resolution to %s', res)
        print('Interpolating resolution to {}'.format(res))
        lons, lats = np.meshgrid(np.arange(lon[0], lon[-1], res), np.arange(
                                                     lat[0], lat[-1]+res, res))
        latlon = basemap.interp(latlon, lon, lat, lons, lats, order=3)
    else:
        lons, lats = np.meshgrid(lon, lat)

    print('plotPreprocess complete')
    return lons, lats, lonlat, latlon, vmin, vmax, latMin, latMax, lonMin, lonMax, label_name

def main(output_dir):
    resolution = 5
    # Read data TODO: read more than one file
    files = np.sort(glob.glob(output_dir[0]+'/AIDA_*.h5'))
    # Selected plotting options TODO: Arguments are different to default, enable argparse options?
    den, nea, neb, lon, lat, alt, what, vmax, tecu = plotOptions(files, mean=False,
         what='tec', tecu=True, group=False,
        difference=True, res=resolution, log=False, neqDiff=False) 
    (lons, lats, lonlat, latlon, vmin, vmax, latMin, latMax, lonMin, lonMax,
        label_name) = plotPreprocess(den, nea, neb, lon, lat, alt, what, res=resolution, tecu=tecu)

    vmin = latlon.min()
    vmax = latlon.max()
    print("latlon min and max is {:.2f} and {:.2f}".format(vmin, vmax))
    cmax = max(abs(vmin), abs(vmax))

    fig = go.Figure(
    #TODO: colorbar scale toggle for log option
        [
            go.Densitymapbox(
                lon = lons.flatten(),
                lat = lats.flatten(),
                z = latlon.flatten(),
                zmin = -cmax, 
                zmax = cmax,
                colorbar = dict(
                    title = label_name,
                    titleside = "right",
                ),
                colorscale = "PuOr_r",
                radius = 30,
                opacity = 0.5,
                
            )
        ]
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center_lon=0,
        margin={"r":0,"t":0,"l":0,"b":0},
        width = 1200,
        height = 700
    )

    app.layout = html.Div(children=[
        html.Nav(
            className = "navbar navbar-expand-md fixed-top",
            children = [
                html.Div(
                    className = "navbar-brand",
                    children = ["AIDA Dashboard"],
                    style = {"font-size": 36, "padding": "20px 20px 20px 20px"}
                )
            ]
        ),
        html.Div(
            dcc.Markdown(
                """
                A web application for visualising AIDA simulation outputs.
                """,
                style={"padding": "35px 35px 20px 20px",
                    "font-family": "sans-serif"
                },
            )
        ),
        dcc.Graph(
            id='graph-tec',
            figure=fig,
            style={"padding": "0 35px 20px 20px",    
                },            
        )
    ])

    print("main complete")

    # return den, nea, neb, lons, lats, lonlat, latlon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create AIDA web dashboard.')
    parser.add_argument('output_dir', metavar='output_dir', type=str, nargs=1,
                        help='specify output directory containing AIDA .h5 files')
    args = parser.parse_args()
    # den, nea, neb, lons, lats, lonlat, latlon = main(args.output_dir)
    main(args.output_dir)
    app.run_server(debug=True)