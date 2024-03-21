

## IMPORTS

# generic
import numpy as np
from datetime import datetime
import xarray as xr

## plotting modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
# cartopy mapping
import cartopy.crs as ccrs
import cartopy.feature as cfeature


## METHODS

def xyskip_for_quiver(lats,lons,n_arrows=23):
    """
    return xskip,yskip to get approx n_arrows for a quiver plot using x[::xskip],y[::yskip]
    """
    xskip=int(np.max([len(lons)//n_arrows-1,1]))
    yskip=int(np.max([len(lats)//n_arrows-1,1]))
    return xskip,yskip

def quiverwinds(lats,lons,u,v,
                thresh_windspeed=10,
                n_arrows=15,
                no_defaults=False,
                add_quiver_key=True,
                differential=False,
                **quivargs):
    """
    u, v in metres/second
    arrows and labels converted into km/h
    ARGS:
        lats,lons,u,v,
        thresh_windspeed=10 km/h
        differential=False: set to True to instead plot quiver difference from mean flow
    """
    #set some defaults
    if (not no_defaults) and (not differential):
        if "scale_units" not in quivargs.keys() and 'scale' not in quivargs.keys():
            # I think this makes 50 m/s 1 inch 
            quivargs['scale_units']="inches"
            quivargs['scale']=50 * 3.6
        if "pivot" not in quivargs.keys():
            quivargs['pivot']='middle'
    
    # xskip and yskip for visibility of arrows
    if n_arrows is not None:
        xskip,yskip=xyskip_for_quiver(lats,lons,n_arrows=n_arrows)
    else:
        xskip,yskip=1,1
    
    # may use 2d lats/lons
    if len(lats.shape)==2:
        qlats=lats[::yskip,::xskip]
        qlons=lons[::yskip,::xskip]
    else:
        qlons=lons[::xskip]
        qlats=lats[::yskip]
    
    # if we want to look at differential wind speed we need to subtract the mean flow
    # also change the wind threshold etc...
    if differential:
        ubar = np.nanmean(u)
        vbar = np.nanmean(v)
        u_diff = (u[::yskip,::xskip] - ubar)*3.6  # m/s -> km/h
        v_diff = (v[::yskip,::xskip] - vbar)*3.6  # m/s -> km/h
        diffstr = "(u-%.1f,v-%.1f)\n"%(ubar*3.6,vbar*3.6)
        Q = plt.quiver(qlons,qlats,
                u_diff,
                v_diff,
                **quivargs)
        plt.quiverkey(Q, 0.1, 1.06, 20, diffstr + r' = $20 \frac{km}{h}$', labelpos='W', fontproperties={'weight':'bold','size':'8'})
        return
        
    # wins speed used as threshhold for quiver arrows
    qs = np.sqrt(u[::yskip,::xskip]**2+v[::yskip,::xskip]**2) * 3.6 # km/h
    
    # winds below 10m/s are hidden this way
    qu = np.ma.masked_where(qs<thresh_windspeed, 
                            u[::yskip,::xskip]) * 3.6
    qv = np.ma.masked_where(qs<thresh_windspeed, 
                            v[::yskip,::xskip]) * 3.6
    
    ## color quivers above 90km/h
    Q = plt.quiver(qlons, qlats, 
               qu, qv, 
               **quivargs)
    
    if add_quiver_key:
        plt.quiverkey(Q, 0.1, 1.05, 50, r'$50 \frac{km}{h}$', labelpos='W', fontproperties={'weight': 'bold'})
    
    ## overlay quivers above 90km/h in magenta
    qu2 = np.ma.masked_where(qs<90, 
                            u[::yskip,::xskip]) * 3.6
    qv2 = np.ma.masked_where(qs<90, 
                            v[::yskip,::xskip]) * 3.6
    Q2 = plt.quiver(qlons, qlats, 
                qu2, qv2,
                color="magenta",
                **quivargs)

def distance_between_points(latlon0,latlon1):
    """
    return distance IN METRES between lat0,lon0 and lat1,lon1 
    calculated using haversine formula, shortest path on a great-circle
     - see https://www.movable-type.co.uk/scripts/latlong.html
    """
    R = 6371e3 # metres (earth radius)
    lat0, lon0 = latlon0
    lat1, lon1 = latlon1
    latr0 = np.deg2rad(lat0)
    latr1 = np.deg2rad(lat1)
    dlatr = np.deg2rad(lat1-lat0)
    dlonr = np.deg2rad(lon1-lon0)
    a = np.sin(dlatr/2.0)**2 + np.cos(latr0)*np.cos(latr1)*(np.sin(dlonr/2.0)**2)
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c

def latslons_axes_along_transect(lats,lons,start,end,nx):
    """
    INPUTS:
        lats, lons: 1darrays of degrees
        start: [lat0,lon0] # start of transect
        end: [lat1,lon1] # end of transect
    return lons,lats
        interpolated degrees from start to end along lats,lons grid
    """
    x_factor = np.linspace(0,1,nx)
    lat0,lon0 = start
    lat1,lon1 = end
    lons_axis = lon0 + (lon1-lon0)*x_factor
    lats_axis = lat0 + (lat1-lat0)*x_factor
    return lats_axis,lons_axis

def transect_interp(data, lats, lons, start, end, nx=20, z=None,
             interpmethod='linear'):
    '''
    interpolate along cross section
    inputs: 
        data:[[z], lats, lons] array
        lats, lons: horizontal dims 1d arrays
        start = [lat0,lon0]
        end = [lat1,lon1]
        nx = how many points along horizontal.
        z_th = optional altitude array [z, lats, lons]
    RETURNS: 
        struct: {
            'transect': vertical cross section of data 
            'x': 0,1,...,len(X axis)-1
            'y': y axis [Y,X] in terms of z
            'lats': [X] lats along horizontal axis
            'lons': [X] lons along horizontal axis
        } 
        xaxis: x points in metres
        yaxis: y points in metres or None if no z provided
    '''
    lat1,lon1 = start
    lat2,lon2 = end
    
    # Interpolation line is really a list of latlons
    lataxis,lonaxis = latslons_axes_along_transect(lats,lons,start,end,nx=nx)
    # Create label to help interpret output
    label=["(%.2f, %.2f)"%(lat,lon) for lat,lon in zip(lataxis,lonaxis)]
    xdistance = np.array([distance_between_points(start, latlon) for latlon in zip(lataxis,lonaxis)])
    
    # Lets put our data into an xarray data array 
    coords = []
    if len(data.shape) ==3:
        coords = [("z",np.arange(data.shape[0]))]    
    coords.extend([("lats",lats),("lons",lons)])
    da = xr.DataArray(data,
                      coords)
    # we also put lat and lon list into data array with new "X" dimension
    da_lats = xr.DataArray(lataxis,dims="X")
    da_lons = xr.DataArray(lonaxis,dims="X")
    
    # interpolat to our lat,lon list
    slicedata = np.squeeze(da.interp(lats=da_lats,lons=da_lons,method=interpmethod).values)
    X=xdistance
    Y=None
    
    if z is not None:
        NZ=data.shape[0] # levels is first dimension
        da_z = xr.DataArray(z,
                            coords)
        # Y in 2d: Y [y,x]
        Y = np.squeeze(da_z.interp(lats=da_lats,lons=da_lons,method=interpmethod).values)
        # X in 2d: X [y,x]
        X = np.repeat(xdistance[np.newaxis,:],NZ,axis=0)
        
    return {'transect':slicedata, # interpolation of data along transect
            'xdistance':xdistance, # [x] metres from start
            'x':X, # [[y,]x] # metres from start, repeated along z dimension
            'y':Y, # [y,x] # interpolation of z input along transect
            'xlats':lataxis, # [x] 
            'xlons':lonaxis, # [x]
            'xlabel':label, # [x]
            }

def transect_winds_interp(u,v,lats,lons,start,end,nx=20,z=None):
    """
    Get wind speed along arbitrary transect line
    ARGUMENTS:
        u[...,lev,lat,lon]: east-west wind speed
        v[...,lev,lat,lon]: north_south wind speed
        lats[lat]: latitudes
        lons[lon]: longitudes
        start[2]: lat,lon start point for transect
        end[2]: lat,lon end point for transect
        nx: optional number of interpolation points along transect
        z[...,lev,lat,lon]: optional altitude or pressure levels

    RETURNS: structure containing:
        'transect_angle': transect line angle (counter clockwise positive, east=0 degrees)
        'transect_wind':wind along transect (left to right is positive),
        'transect_v':v cross section,
        'transect_u':u cross section,
        'x': metres from start point along transect,
        'y': metres above surface along transect,
        'lats':transect latitudes,
        'lons':transect longitudes,
        'label':[x,'lat,lon'] list for nicer xticks in cross section

    """
    lat0,lon0=start
    lat1,lon1=end
    # signed angle in radians for transect line
    theta_rads=np.arctan2(lat1-lat0,lon1-lon0)
    theta_degs=np.rad2deg(theta_rads)
    #print("CHECK: angle between", start, end)
    #print("     : is ",theta_degs, "degrees?")
    # base interp points on grid size

    ucross_str=transect_interp(u,lats,lons,
                    start=[lat0,lon0],
                    end=[lat1,lon1],
                    nx=nx,
                    z=z)
    ucross = ucross_str['transect']
    vcross_str=transect_interp(v,lats,lons,
                    start=[lat0,lon0],
                    end=[lat1,lon1],
                    nx=nx,
                    z=z)
    vcross = vcross_str['transect']
    wind_mag = ucross * np.cos(theta_rads) + vcross * np.sin(theta_rads)

    ret={
        'transect_angle':theta_degs,
        'transect_wind':wind_mag,
        'transect_v':vcross,
        'transect_u':ucross,
        'x':ucross_str['x'],
        'y':ucross_str['y'],
        'xdistance':ucross_str['xdistance'],
        'xlats':ucross_str['xlats'],
        'xlons':ucross_str['xlons'],
        'xlabel':ucross_str['xlabel'],
        'nx':nx,
        }
    return ret

def plot_transect(data, z, lat, lon, start, end, npoints=20, 
             topog=None, sh=None, latt=None, lont=None, ztop=5000,
             title="", ax=None, colorbar=True,
             lines=None,
             cbar_args={},
             **contourfargs):
    '''
    Draw cross section
    ARGUMENTS:
        data is 3d [levs,lats,lons]
        z (3d): height [lev,lats,lons]
        lat(1d), lon(1d): data coordinates
        start, end: [lat0,lon0], [lat1,lon1]
        lines (list): add black lines to contourf
        cbar_args = dict of options for colorbar, drawn if colorbar==True
    return slicedata, slicex, slicez, cmappable (the colorbar)
    '''
    ## Default contourfargs
    if 'extend' not in contourfargs:
        contourfargs['extend'] = 'max'

    ## Check that z includes topography (within margin of 40 metres)
    if topog is not None:
        if np.mean(z[0]+40)<np.mean(topog):
            print("ERROR:",np.mean(z[0]), np.min(z[0]), "(mean,lowest z) is lower than topog", np.mean(topog), np.min(topog))
            print("ERROR:", "Try adding topog to each level of z")
            assert False
        
    # Transect slice: data[z,x] x[z,x], z[z,x]
    # struct: {
    #         'transect': vertical cross section of data 
    #         'x': x axis [Y,X] in metres from start point 
    #         'y': y axis [Y,X] in terms of z_th
    #         'lats': [X] lats along horizontal axis
    #         'lons': [X] lons along horizontal axis
    #     } 
    transect_struct = transect_interp(data,lat,lon,start,end,nx=npoints, z=z)
    slicedata = transect_struct['transect']
    # X axis [0,1,...]
    X = transect_struct['x']
    # heights along transect [x,y]
    Y = transect_struct['y']
    
    if ax is not None:
        plt.sca(ax)
    # Note that contourf can work with non-plaid coordinate grids provided both are 2-d
    # Contour inputs: xaxis, yaxis, data, colour gradient 
    cmappable=plt.contourf(X,Y,slicedata,**contourfargs)
    
    if colorbar:
        # defaults if not set
        #orientation 	vertical or horizontal
        #fraction 	0.15; fraction of original axes to use for colorbar
        #pad 	0.05 if vertical, 0.15 if horizontal; fraction of original axes between colorbar and new image axes
        #shrink 	1.0; fraction by which to multiply the size of the colorbar
        #aspect 	20; ratio of long to short dimensions
        if 'pad' not in cbar_args:
            cbar_args['pad']=0.01
        if 'aspect' not in cbar_args:
            cbar_args['aspect']=30
        if 'shrink' not in cbar_args:
            cbar_args['shrink'] = 0.85
        if 'fraction' not in cbar_args:
            cbar_args['fraction'] = .075
        plt.colorbar(**cbar_args)
    
    # Add contour lines
    if lines is not None:
        plt.contour(X,Y,slicedata,lines,colors='k')
    
    zbottom = np.tile(np.min(Y),reps=npoints) # xcoords
    xbottom = X[0,:]
    # make sure land is obvious
    if topog is not None:
        # Pull out cross section of topography and height
        if latt is None:
            latt=lat
        if lont is None:
            lont=lon
        slicetopog_struct = transect_interp(topog,latt,lont,start,end,nx=npoints)
        slicetopog = slicetopog_struct['transect']
        # Plot gray fill for topography
        plt.fill_between(xbottom, slicetopog, zbottom, 
                         interpolate=True, facecolor='darkgrey',
                         zorder=2)
            
    if ztop is not None:
        plt.ylim(0,ztop)
    
    plt.xticks([])
    plt.xlabel('')
    plt.title(title)

    return slicedata, X, Y, cmappable

def wind_cross_section(U, V, Z, lats, lons, start, end, topog=None, n_X=20, n_arrows=10, ztop=5000, W=None, add_quiver=True):
    """ transect along line, X will be horizontal, Y vertical 
    ARGS:
        u,v,w,z [lev,lats,lons]: u,v,w is m/s winds, z is level altitudes in m
        start = [lat0,lon0],
        end = [lat1,lon1],
        topog: [lats,lons] array of topography heights
        ztop: top altitude default 5000m
        n_X(optional int): how many points along x axis do we interpolate to
    """
    retdict = {}
    
    # First we subset all the arrays so to be below the z limit
    zmin = np.min(Z, axis=(1,2)) # lowest altitude on each model level
    ztopi = np.argmax(ztop<zmin)+1 # highest index where ztop is less than model level altitude
    u,v,z = [U[:ztopi],V[:ztopi],Z[:ztopi]]
    
    # transect direction left to right wind speed
    transect_winds_struct = transect_winds_interp(u,v,lats,lons,start,end,nx=n_X,z=z)
    transect_s = transect_winds_struct['transect_wind']
    slicex = transect_winds_struct['x']
    slicez = transect_winds_struct['y']
    
    # Need to account for vector stretching
    Yscale=transect_winds_struct['xdistance'][-1] / ztop
    
    print("DEBUG: yscale=%.2f (transect length/height)"%Yscale,"   mean horizontal wind speed=",np.nanmean(transect_s))
    
    ## contourf our horizontal wind speeds on the transect
    cmappable=plt.contourf(slicex,slicez,transect_s)
    
    # make sure land is obvious
    if topog is not None:
        zbottom = np.tile(np.min(slicez),reps=n_X) # xcoords
        xbottom = slicex[0,:]
        # Pull out cross section of topography and height
        slicetopog_struct = transect_interp(topog,lats,lons,start,end,nx=n_X)
        slicetopog = slicetopog_struct['transect']
        # Plot gray fill for topography
        plt.fill_between(xbottom, slicetopog, zbottom, 
                         interpolate=True, facecolor='darkgrey',
                         zorder=2)
    
    if add_quiver:
        ## Y scale applied to vertical motion adjusts the arrows so that they scale with plot dimensions
        # for instance, if the transect is 50km long, and 5km tall, a wind moving 5m/s horizontally and 5m/s upwards is not really pointing 45 degrees upwards
        # it should instead point nearly straight up as it will leave the plot through the roof way before it reaches the end of the 50km transect
        # vertical wind speed along transect
        w = W[:ztopi]
        transect_w_struct = transect_interp(w,lats,lons,start,end,nx=n_X,z=z)
        transect_w = transect_w_struct['transect']
        quiverwinds(slicez,slicex, # z axis, x axis, 
                    transect_s,transect_w * Yscale,
                    n_arrows=n_arrows,
                    add_quiver_key=False,
                    alpha=0.5,
                    )
        # need to reset limits if we added quivers
        plt.xlim(np.nanmin(slicex),np.nanmax(slicex))
        plt.ylim(np.nanmin(slicez),ztop)
    
    plt.colorbar(cmappable)