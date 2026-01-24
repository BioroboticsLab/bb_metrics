import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cv2

snscolors=sns.color_palette()
snscolors = np.concatenate((snscolors,snscolors))


# This needs to be called and set just after import, and before using the package.  It sets which version of 'definitions' to use
def init(bd_input):  
    global bd
    bd = bd_input


#  creates a rgba image from histogram data, using a specified color and normalization
def rgba_cmap(histdata,normvalue=-1,color=[0,0,0]):
    new = np.zeros((histdata.shape[0],histdata.shape[1],4))
    if normvalue<0:
        normvalue = np.max(histdata)
    vals = 1 - np.minimum(histdata/normvalue,1)
    
    new[:,:,0] = color[0]
    new[:,:,1] = color[1]
    new[:,:,2] = color[2]
    new[:,:,3] = 1-vals
    return new

import numpy as np

def setimageorientation(f, ax, setf=True):
    f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)            
    
    # Check if ax is a single axis or an array of axes
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    # If ax is an array, iterate through each axis
    for a in ax.flat:  # .flat flattens the axes array, works for 2D/1D axes arrays
        a.set_xlim([0, 2 * bd.xpixels])
        a.set_ylim([0, bd.ypixels])

def createnewimage(size=10):
    f, ax = plt.subplots(1,1)
    f.set_size_inches(size,size*5312/9216)
    setimageorientation(f,ax)
    # ax.set_ylim([bd.ypixels,0])  # note this is backwards.. this makes it show 'zero' at the top, which is what shows for the comb, so its consistent (and no changes to data)    
    # ax.invert_yaxis()
    return f, ax

def showimages(images, ax=None, alpha=1, downsample=1):
    """
    Plots a set of images, optionally downsampling them while maintaining the original pixel coordinates.

    Parameters:
    - images: List of 2D numpy arrays representing the images to be plotted.
    - ax: Optional matplotlib axis. If None, a new figure is created.
    - alpha: Transparency level for the image plot.
    - downsample: Integer factor to downsample the images. Must be >= 1. Default is 1 (no downsampling).

    Returns:
    - ax: The matplotlib axis with the plotted images.
    """
    if downsample < 1:
        raise ValueError("Downsample factor must be >= 1.")
    
    if ax is None:
        f, ax = createnewimage()
    
    # Downsample each image
    downsampled_images = [
        img[::downsample, ::downsample] for img in images
    ]
    
    # Stack downsampled images vertically
    combined_image = np.vstack(downsampled_images).T

    # Create extents to map downsampled image space to original image space
    original_width, original_height  = images[0].shape
    extent = [0, original_width*2, original_height, 0]  # [xmin, xmax, ymin, ymax]

    # Plot the downsampled image with adjusted extents
    ax.imshow(combined_image, cmap='gray', alpha=alpha, extent=extent)

    # Hide axis ticks
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    return ax
    

def showhist(hist,ax=[],color=[0,0,0],alpha=1,normvalue=-1):
    if ax==[]:
        f, ax = createnewimage()
    if normvalue==-1:
        normvalue=np.quantile(hist,0.99)

    rgba_img = rgba_cmap(hist.T,normvalue=normvalue,color=color)
    ax.imshow(rgba_img,extent=(0,2*bd.xpixels,bd.ypixels,0),alpha=alpha,rasterized=True)
#     ax.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
def showmonthday(timestamp):
    return str(timestamp.month).zfill(2)+'-'+str(timestamp.day).zfill(2)
        
def showframe(cam_ids,ax=[],color=snscolors[1]):  # cam_ids should either be [0,1], or [2,3], for the two different hives
    if ax==[]:
        f, ax = createnewimage()        
    for v in [0, bd.xpixels, 2*bd.xpixels]:
        ax.axvline(bd.xpixels,c=color,zorder=10)
    for i,cam_id in enumerate(cam_ids):
        ax.axhline(bd.divs_middle[cam_id],xmin=i*0.5,xmax=0.5+i*0.5,c=color)
    return ax

def get_hive_cam0(camera):
    import numbers
    # If camera is a single integer (int or int-like, such as int64), convert it to a list
    if isinstance(camera, numbers.Integral):
        camera = [camera]
    
    # Check for any 0 or 1 in the camera list
    if any(cam in [0, 1] for cam in camera):
        return 0
    # Check for any 2 or 3 in the camera list
    elif any(cam in [2, 3] for cam in camera):
        return 2
    else:
        return None  # Or any other default value

def plotbee_xy(
    dfbee, plot_tracklets=False, ax=None, color='k', s=10, alpha=0.7, alpha_untagged=0.7, joined=True, maxxydiff=80, rasterized=False
):
    """
    Plots adjusted coordinates (shifted x) to show all points for both sides.
    Optionally splits trajectories by `track_id` and plots lines for each segment.

    Parameters:
    - dfbee: DataFrame with columns 'x_pixels', 'y_pixels', 'cam_id', and optionally 'track_id'.
    - ax: Optional matplotlib axis. If None, a new figure is created.
    - color: Color of the plot. Can be a single value or list for multiple trajectories.
    - s: Size of the plot markers or line thickness.
    - alpha: Transparency level.
    - joined: If True, joins the points with a line; else scatter plot.
    - maxxydiff: Maximum allowed difference in x and y coordinates for joining.
    - rasterized: Boolean to enable rasterization for the plot.
    """
    if ax is None:
        f, ax = createnewimage()  # Create a new plot if no axis is provided

    # Ensure that the relevant columns exist in dfbee
    if len(dfbee) > 0 and {'x_pixels', 'y_pixels', 'cam_id'}.issubset(dfbee.columns):
        dfbee = dfbee.copy()
        x = dfbee['x_pixels'].to_numpy(float)
        y = dfbee['y_pixels'].to_numpy(float)
        camera = dfbee['cam_id'].to_numpy(int)
        # Adjust x based on the camera
        dfbee['x_adjusted'] = x + (camera - get_hive_cam0(dfbee['cam_id'])) * bd.xpixels  # Camera 0 left, camera 1 on the right
        if joined:
            if plot_tracklets:
                # Group by 'track_id' only
                grouped = dfbee.groupby('track_id')

                for track_id, group in grouped:
                    group_x_adjusted = group['x_adjusted'].to_numpy(float)
                    group_y = group['y_pixels'].to_numpy(float)
                    group_camera = group['cam_id'].to_numpy(int)
                    
                    # Calculate differences between consecutive points
                    xydiff = np.sqrt(np.diff(group_x_adjusted)**2 + np.diff(group_y)**2)
                    # Split condition for joining lines
                    splitcond = np.where(
                        (xydiff > maxxydiff) | (np.abs(np.diff(group_camera)) > 0)
                    )[0] + 1

                    # Split the trajectory based on the condition
                    xtp = np.split(group_x_adjusted, splitcond)
                    ytp = np.split(group_y, splitcond)

                    # Plot the joined trajectories
                    for xt, yt in zip(xtp, ytp):
                        if len(xt) > 1:
                            if np.any(group['bee_id'].isna()):
                                  # cycle colors for untagged bees
                                ax.plot(xt, yt, color=snscolors[int(track_id)%10], alpha=alpha_untagged, linewidth=s / 10, rasterized=rasterized)
                            else:
                                ax.plot(xt, yt, color=color, alpha=alpha, linewidth=s / 10, rasterized=rasterized)
                            
            else:
                # No 'track_id' column; proceed as before
                # Calculate differences between consecutive points
                xydiff = np.sqrt(np.diff(dfbee['x_adjusted'])**2 + np.diff(y)**2)
                # Split condition for joining lines
                splitcond = np.where(
                    (xydiff > maxxydiff) | (np.abs(np.diff(camera)) > 0)
                )[0] + 1

                # Split the trajectory based on the condition
                xtp = np.split(dfbee['x_adjusted'].to_numpy(), splitcond)
                ytp = np.split(y, splitcond)

                # Plot the joined trajectories
                for xt, yt in zip(xtp, ytp):
                    if len(xt) > 1:
                        ax.plot(xt,yt,color=color,alpha=alpha,linewidth=s / 10, rasterized=rasterized )
        else:
            # Plot as scatter points if not joined
            ax.scatter(dfbee['x_adjusted'], y, color=color, s=s, alpha=alpha, rasterized=rasterized)
    return ax


# ────────────────────────────────────────────────────────────────────
# Feeder plotting helpers (adapted from Read feeder spreadsheet.ipynb)
# These are used by downstream notebooks to visualize per-video counts.
# ────────────────────────────────────────────────────────────────────

def plot_feedercam_segments(ax, df_cam, color, gap_seconds=32, whichcol='totalcounts'):
    """
    Plot a line per contiguous segment, breaking when timestamp gaps exceed gap_seconds.
    df_cam must include ['timestamp_start', whichcol, 'TaggedBee'] and optional 'hive'.
    """
    if df_cam.empty:
        return ax

    df = df_cam.sort_values('timestamp_start').reset_index(drop=True)
    df['new_seg'] = df['timestamp_start'].diff() > pd.Timedelta(seconds=gap_seconds)
    df['seg_id'] = df['new_seg'].cumsum().fillna(0).astype(int)

    first = True
    for _, grp in df.groupby('seg_id'):
        if whichcol == 'TaggedBee':
            ax.plot(
                grp['timestamp_start'], grp['TaggedBee'],
                color=color, linestyle='--',
                label='TaggedBee' if first else "", alpha=0.8
            )
        else:
            ax.plot(
                grp['timestamp_start'], grp[whichcol],
                color=color, linestyle='-',
                label=whichcol if first else "", alpha=0.8
            )
        first = False

    ax.set_ylabel("Count")
    ax.legend()
    return ax


def shade_treatments(ax, intervals_df, color='red', alpha=0.3):
    """Shade treatment intervals on a given Matplotlib axis; expects columns ['start','end'].""" 
    if intervals_df is None or len(intervals_df) == 0:
        return ax
    for _, row in intervals_df.iterrows():
        ax.axvspan(row['start'], row['end'], color=color, alpha=alpha)
    return ax


def common_plot_formatting(ax_list, mintime, maxtime, tz='Europe/Berlin'):
    """
    Apply shared x-axis locators/formatting across a list of axes.
    - Major ticks: daily grid lines
    - Minor ticks: every 6 h, labels like '12:00'
    """
    daily_locator = mdates.DayLocator(tz=tz)
    day_fmt = mdates.DateFormatter('%m-%d', tz=tz)
    sixh_locator = mdates.HourLocator(byhour=range(0, 24, 6), tz=tz)
    hour_fmt = mdates.DateFormatter('%H:%M', tz=tz)

    for i, a in enumerate(ax_list):
        a.xaxis.set_major_locator(daily_locator)
        a.xaxis.set_major_formatter(day_fmt)
        a.xaxis.set_minor_locator(sixh_locator)
        a.xaxis.set_minor_formatter(hour_fmt)
        a.xaxis.set_tick_params(which='major', labelbottom=False)
        a.xaxis.set_tick_params(which='minor', labelbottom=True, labelsize=10, rotation=90)
        a.grid(axis='x', which='major', color='gray', linestyle='--', linewidth=0.7)

        if i == 0:
            start_day = pd.Timestamp(mintime.date() + pd.Timedelta(days=1), tz=tz)
            end_day = pd.Timestamp(maxtime.date(), tz=tz)
            for day in pd.date_range(start_day, end_day, freq='D', tz=tz):
                noon = day + pd.Timedelta(hours=14)
                a.annotate(day.strftime('%m-%d'), xy=(noon, 1.02), xycoords=('data', 'axes fraction'),
                           ha='center', va='bottom', fontsize=12)

    ax_list[-1].set_xlim(mintime, maxtime)
    for a in ax_list:
        a.set_xlabel('')
        a.tick_params(labelsize=12)


def plot_temp_and_precip(a, hourdata):
    """Plot temp on left y and precip on right y."""
    a.plot(hourdata.index, hourdata['temp'], marker='.', color='grey')
    a.set_ylabel('Outside\\nTemperature (°C)', fontsize=14)
    ax2 = a.twinx()
    ax2.bar(hourdata.index, hourdata['prcp'], width=0.05, color='tab:blue', alpha=0.6, label='Precipitation (mm)')
    ax2.set_ylabel('Precipitation (mm)', color='tab:blue', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='tab:blue')


def plot_sun_thirdaxis(a, hourdata):
    """Add sunshine (minutes/hr) on a third y-axis."""
    ax3 = a.twinx()
    ax3.spines['right'].set_position(("axes", 1.035))
    ax3.plot(hourdata.index, hourdata['tsun'], color='orange', linestyle='-', label='Sunshine (min/hr)', alpha=0.7)
    ax3.set_ylabel('Sunshine (min/hr)', fontsize=14, color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')





###### DISPLAY - PCA related ##################################
############################################################

def pcacomponentplots(ax,vh,ylabels,colors=''):
    numev = vh.shape[0]
    xlim = 1.05*np.max(np.abs(vh))
    if colors=='':
        colors = np.tile([0.35,0.35,0.35],(numev,1))
    elif len(colors)<numev:
        colors = np.tile(colors,(numev,1))
    for evnum in range(numev):
        a=ax[evnum]
        x=vh[evnum]
        y = np.flipud(np.arange(vh.shape[1]))
    #     ax.plot(x,y,'-o',label='$\\vec e_'+str(evnum)+'$: '+str(np.round(pcavar[evnum]*100,1))+'%',c=snscolors[evnum])
        thickness=0.75
        a.barh(y+0*(evnum-1)*thickness,x,height=thickness,color=colors[evnum])
        a.set_yticks(y)
        a.set_yticklabels(ylabels,rotation='horizontal',fontsize=12)
        a.tick_params(labelsize=12)
        a.axvline(0,c='k',linestyle='-',linewidth=1)
#         a.set_title(label='$\\vec v_'+str(evnum)+'$: '+str(np.round(pcavar[evnum]*100,1))+'%',fontsize=14)
        a.set_xlim([-xlim,xlim])
        a.set_ylim([-0.5,len(y)-0.5])

def plot_tsne_withcolors(ax,tsne_result,quantity,title,corrskip=1,plotskip=1,colortype='scalar',qmin=0.001,qmax=0.999,alphaval=0.3,s=4,coloroffset=0,cmapname='cool',setxylimquantile=False):
    colordata = quantity.copy()
    if len(colordata)>len(tsne_result):
        colordata = colordata[::corrskip]
    if len(colordata.shape)>1:
        colordata = colordata[:,0]
    if colortype=='scalar':
        cmap=plt.get_cmap(cmapname)  # or 'cool'
        q0,q1 = np.quantile(colordata,[qmin,qmax])
        colordata = colordata-q0
        colordata = colordata/(q1-q0)
        colordata[colordata<0] = 0
        colordata[colordata>1] = 1
        colordata *= 0.99
        colors = cmap(colordata)
    else:
#         cmap=plt.get_cmap('Set1')
        colors = snscolors[colordata.astype(int)+coloroffset]
    tp = tsne_result
    # [ax.scatter([-100],[-100],alpha=1,s=10,color=cmap(i*0.99/np.max(groupvalues)),label='group '+str(i)) for i in np.arange(max(groupvalues)+1)]  # legend hack
    scatterplot = ax.scatter(tp[::plotskip,0],tp[::plotskip,1],s=s,alpha=alphaval,color=colors[::plotskip],rasterized=True)
    if setxylimquantile:
        ax.set_xlim(np.quantile(tp[:,0],[qmin,qmax]))
        ax.set_ylim(np.quantile(tp[:,1],[qmin,qmax]))
    else:
        ax.set_xlim(np.quantile(tp[:,0],[0,1]))
        ax.set_ylim(np.quantile(tp[:,1],[0,1]))
    ax.set_title(title,fontsize=16)       
    return scatterplot, colordata    
    
def categorydists(n_clusters,membership,quantityvals,labels,pointskip=1,coloroffset=0,ax='',f='',npoints_in_title=False):
    numq = len(quantityvals)
    if len(ax)==0:
        f,ax = plt.subplots(1,n_clusters,sharex=True,sharey=True)
        f.set_size_inches(3*n_clusters,3.2)

    for i in range(n_clusters):
        a = ax[i]
        clr = snscolors[i+coloroffset] if coloroffset>=0 else 'k'
        sel = membership == i
        a.set_title('Cluster '+str(i+1)+(': '+str(np.sum(sel).astype(int))+' data points' if npoints_in_title else ''),fontsize=14)
        for j, q in enumerate(quantityvals):
            tp = q[sel]
            alpha_scaled = 0.2
            xval = len(quantityvals)-j-1
            bplot = a.boxplot(x=tp,positions=[xval],patch_artist=True,showfliers=False,showcaps=True,vert=False,widths=0.9)
            for patch in bplot['boxes']:
                patch.set(color=clr,alpha=alpha_scaled)  
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bplot[element], color=clr,alpha=1)     
            plt.setp(bplot["fliers"], markeredgecolor=clr, markerfacecolor=clr,markersize=4,alpha=alpha_scaled)
            xnoise=0.1
            xtp = np.random.normal(xval,xnoise,size=len(tp))
            a.scatter(tp[::pointskip],xtp[::pointskip],color=clr,alpha=0.03,zorder=10,s=3,rasterized=True)
    for a in ax:
        a.set_yticks(range(len(quantityvals)))
        a.set_yticklabels(np.flip(labels),rotation='horizontal',fontsize=12)
        a.axvline(0,c='k',linewidth=1)
        a.set_xlabel('Quantity (std. dev from mean)',fontsize=12)
        a.set_ylim([-0.5,len(quantityvals)-0.5])
    return f,ax

def quantitydists(n_clusters,membership,quantityvals,labels,f='',ax='',pointskip=1,coloroffset=0,xorder=[],colorsel=[],color='k'):
    numq = len(quantityvals)
    if len(ax)==0:
        f,ax = plt.subplots(1,numq,sharex=True,sharey=True)
        f.set_size_inches(2*numq,3)
        if numq==1:
            ax = np.array([ax])

    if len(xorder)==0:
        xorder = np.arange(n_clusters)
    if len(colorsel)==0:
        colorsel=np.arange(n_clusters)
    # dmat_bygroup = [[ for n in range(n_clusters)] for qnum in range(dmat.shape[1])]
    for i in range(numq):
        a = ax[i]
        a.set_title(labels[i],fontsize=14)
        for j in range(n_clusters):
            if coloroffset>=100:
                clr = (color[j] if len(color)==n_clusters else color)
            else:
                clr = snscolors[colorsel[j]+coloroffset]
            tp = quantityvals[i][membership==j]
            alpha_scaled = 0.2
            xval = xorder[j]
            bplot = a.boxplot(x=tp,positions=[xval],patch_artist=True,showfliers=False,showcaps=True,vert=True,widths=0.9)
            for patch in bplot['boxes']:
                patch.set(color=clr,alpha=alpha_scaled)  
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bplot[element], color=clr,alpha=1)     
            plt.setp(bplot["fliers"], markeredgecolor=clr, markerfacecolor=clr,markersize=4,alpha=alpha_scaled)
            xnoise=0.1
            xtp = np.random.normal(xval,xnoise,size=len(tp))
            a.scatter(xtp[::pointskip],tp[::pointskip],color=clr,alpha=0.03,zorder=10,s=3,rasterized=True)
    [a.set_xticks(range(n_clusters)) for a in ax]
    [a.set_xticklabels(np.arange(n_clusters)+1,fontsize=14) for a in ax]
    [a.tick_params(labelsize=14) for a in ax]
    ax[0].set_ylabel('Quantity value',fontsize=14)
    [a.set_xlabel('Cluster number',fontsize=14) for a in ax]
    [a.axhline(0,c='k',linewidth=1) for a in ax]
    return f, ax



###### Weather and temperature sensors ##################################
#########################################################################

## edited from bb_monitor

def plot_temp_and_precip_daily(a, weatherdata):
    """
    Plots daily temperature and precipitation data with temperature as a 
    fill_between plot and precipitation as a bar plot.
    
    Args:
        a: The primary axis for temperature.
        weatherdata: DataFrame containing daily weather data.
    """
    # Plot temperature with fill_between for tmin to tmax and a line for tavg
    a.fill_between(weatherdata.index, weatherdata['tmin'], weatherdata['tmax'], 
                   color='lightgrey', alpha=0.6, label='Min-Max Temp (°C)')
    a.plot(weatherdata.index, weatherdata['tavg'], color='black', linestyle='-', label='Avg Temp (°C)')
    a.set_ylabel('Temperature (°C)', fontsize=14)
    
    # Create a secondary y-axis for the precipitation
    ax2 = a.twinx()
    ax2.bar(weatherdata.index, weatherdata['prcp'], width=0.5, color='tab:blue', alpha=0.6, label='Precipitation (mm)')
    ax2.set_ylabel('Precipitation (mm)', color='tab:blue', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    a.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

def plot_sunshine_duration(a, weatherdata, is_hourly=False):
    """
    Plots the sunshine duration (tsun) on a given axis. 
    
    Args:
        a: The primary axis for plotting.
        weatherdata: DataFrame containing weather data with 'tsun' column.
        is_hourly: Boolean to indicate if the data is hourly. Default is False (for daily data).
    """
    if is_hourly:
        # Plot sunshine duration for hourly data
        a.plot(weatherdata.index, weatherdata['tsun'], color='gold', linestyle='-', label='Sunshine (hours)')
        a.set_ylabel('Sunshine\nDuration (minutes per hour)', fontsize=14)
    else:
        # Plot sunshine duration for daily data
        a.bar(weatherdata.index, weatherdata['tsun']/60, width=0.5, color='gold', alpha=0.8, label='Sunshine (hours)')
        a.set_ylabel('Sunshine\nDuration (hours)', fontsize=14)
        
    a.legend(loc='upper left', fontsize=12)


def plot_temperature_sensors(a,combined_data,label_dict):
    snscolors = sns.color_palette()
    # Plot each temperature sensor data with appropriate colors and line styles
    for hex_code, label in label_dict.items():
        hive = label[0]
        location = label.split()[0]
        if hive == 'A':
            color = snscolors[0]
        elif hive == 'B':
            color = snscolors[1]
            
        if 'honey' in label:
            linestyle = '-'
        elif 'brood' in label:
            linestyle = ':'
        elif 'room' in label:
            linestyle = '--'
    
        a.plot(combined_data['Time'], combined_data[hex_code], label=label, color=color, linestyle=linestyle)
    a.legend(fontsize=10, loc='lower left', ncol=3, borderpad=0.2, columnspacing=1, labelspacing=0.1,handlelength=1.5)
    a.set_ylabel('Hive/Inside\nTemperature (°C)', fontsize=14)  


### COPIED FROM BB_MONITOR
import json

def plot_temp_and_precip_hourly(a, weatherdata):   
    a.plot(weatherdata.index, weatherdata['temp'], color='grey')
    a.set_ylabel('Outside\nTemperature (°C)', fontsize=14)
    ax2 = a.twinx()  # Secondary y-axis for precipitation
    ax2.bar(weatherdata.index, weatherdata['prcp'], width=0.05, color='tab:blue', alpha=0.6, label='Precipitation (mm)')
    ax2.set_ylabel('Precipitation (mm)', color='tab:blue', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='tab:blue')


def plot_feedercam_counts(a, dfcounts,feedercamhives, minutes_to_avg=15, col_to_plot='totalcounts'):
    # Group by 'camID' and resample by 'video_start_timestamp' with the specified interval
    if 'camID' in dfcounts.columns:  # account for change in formatting, and old vs updated name
        camcol = 'camID'
    else:
        camcol = 'cam_id'
    if len(dfcounts)>0:
        resampled_df = dfcounts.groupby(camcol).resample(f'{minutes_to_avg}min', on='video_start_timestamp').mean().reset_index()
        for camID, group in resampled_df.groupby(camcol):
            a.plot(group['video_start_timestamp'], group[col_to_plot], marker='.', linestyle='-', label=feedercamhives[camID])
    a.set_ylabel('Avg counts at feeder', fontsize=14)
    a.legend(title='Hive',fontsize=12,loc=2,title_fontsize=12)


# ────────────────────────────────────────────────────────────────────
# Comb annotation plotting helpers
# ────────────────────────────────────────────────────────────────────

def normalize_label(label: str, keep_labels: set, other_label: str) -> str:
    """Normalize label to known labels or 'other'."""
    return label if label in keep_labels else other_label


def hex_to_rgba(hex_color: str):
    """Convert hex color to RGBA tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 8:  # RRGGBBAA
        r, g, b, a = h[0:2], h[2:4], h[4:6], h[6:8]
        return tuple(int(v, 16) / 255 for v in (r, g, b, a))
    if len(h) == 6:  # RRGGBB
        r, g, b = h[0:2], h[2:4], h[4:6]
        return tuple(int(v, 16) / 255 for v in (r, g, b)) + (1.0,)
    return (1.0, 0.0, 1.0, 1.0)  # fallback magenta


def with_alpha(rgba, a):
    """Return RGBA tuple with modified alpha."""
    return (rgba[0], rgba[1], rgba[2], a)


def plot_annotated_row(
    row,
    label_color_hex,
    label_order,
    keep_labels,
    other_label,
    alpha_fill=0.25,
    alpha_edge=0.1,
    fill_empty_cell=False,
    empty_label="empty_cell",
    rotate_clockwise=True,
    cmap="gray",
    figsize=(10, 7),
    ax=None,
    show=True,
    rgba_grid=None,
    raw_w=None,
    raw_h=None,
    print_debug=False,
):
    """
    Plot annotated comb image with cell labels.

    Can work in two modes:
    1. From annotations JSON (default): plots circles for each annotated cell
    2. From grid (if rgba_grid is provided): overlays a downsampled grid

    Parameters
    ----------
    row : pd.Series
        Row with 'path' (image) and optionally 'annotation_path', 'cam', 'day'
    label_color_hex : dict
        Maps label names to hex colors
    label_order : list
        Preferred ordering of labels for legend
    keep_labels : set
        Labels to keep as-is; others become other_label
    other_label : str
        Label for unmapped categories
    alpha_fill : float
        Alpha for circle fill (annotation mode)
    alpha_edge : float
        Alpha for circle edge (annotation mode)
    fill_empty_cell : bool
        Whether to fill/show empty cells. In annotation mode, controls circle fill.
        In grid mode, if False, sets empty_label pixels to transparent.
    empty_label : str
        Label indicating empty cells. Used in both annotation and grid modes.
    rotate_clockwise : bool
        Whether to rotate image 90° clockwise
    cmap : str
        Colormap for background image
    figsize : tuple
        Figure size if creating new figure
    ax : matplotlib.axes.Axes, optional
        Axis to plot on; if None, creates new figure
    show : bool
        Whether to call plt.show()
    rgba_grid : np.ndarray, optional
        If provided, use grid mode instead of annotations
        Shape: (grid_h, grid_w, 4) with RGBA values
    raw_w : int, optional
        Raw image width (required if rgba_grid is provided)
    raw_h : int, optional
        Raw image height (required if rgba_grid is provided)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    from pathlib import Path
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    import json

    # Get rotation config if available
    try:
        from . import get_config
        from .rotation import get_rotation_config
        cfg = get_config()
        rot_cfg = get_rotation_config(cfg)
        use_rotation_config = True
    except Exception:
        rot_cfg = None
        use_rotation_config = False

    img_path = Path(row["path"])
    img = plt.imread(img_path)
    h_orig, w_orig = img.shape[:2]

    if print_debug:
        print("=" * 70)
        print(f"DEBUG: plot_annotated_row")
        print("=" * 70)
        print(f"Image path: {img_path.name}")
        print(f"Original image shape (loaded): h={h_orig}, w={w_orig}")
        print(f"use_rotation_config: {use_rotation_config}")
        print(f"rotate_clockwise param: {rotate_clockwise}")
        if use_rotation_config and rot_cfg is not None:
            print(f"Rotation config found:")
            print(f"  - rotation type: {rot_cfg.rotation}")
            print(f"  - numpy_rot90_k: {rot_cfg.numpy_rot90_k()}")
            print(f"  - original_width: {rot_cfg.original_width}")
            print(f"  - original_height: {rot_cfg.original_height}")

    # Apply rotation for display
    rotation_applied = False
    rotation_k = 0
    if use_rotation_config and rot_cfg is not None:
        k = rot_cfg.numpy_rot90_k()
        if k != 0:
            img = np.rot90(img, k=k)
            rotation_applied = True
            rotation_k = k
    elif rotate_clockwise:
        # Fallback for backward compatibility
        img = np.rot90(img, k=-1)
        rotation_applied = True
        rotation_k = -1

    if print_debug:
        print(f"Rotation applied: {rotation_applied}")
        if rotation_applied:
            print(f"  - numpy rot90 k value: {rotation_k}")
            print(f"  - rotated image shape: h={img.shape[0]}, w={img.shape[1]}")
        else:
            print(f"  - no rotation applied")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(img, cmap=cmap)

    # Determine which mode to use
    if rgba_grid is not None:
        # Grid mode: overlay the rgba_grid
        if raw_w is None or raw_h is None:
            raise ValueError("raw_w and raw_h must be provided when using rgba_grid")

        # Make a copy to avoid modifying the input
        rgba_grid_copy = rgba_grid.copy()

        # If fill_empty_cell is False, set empty_label pixels to transparent
        if not fill_empty_cell and empty_label in label_color_hex:
            empty_color = hex_to_rgba(label_color_hex[empty_label])
            # Find pixels matching the empty_label color
            empty_mask = np.all(np.abs(rgba_grid_copy[:,:,:3] - empty_color[:3]) < 0.01, axis=2)
            # Set alpha to 0 for those pixels
            rgba_grid_copy[empty_mask, 3] = 0.0

        # Apply same rotation to grid as to image
        if use_rotation_config and rot_cfg is not None:
            k = rot_cfg.numpy_rot90_k()
            rgba_grid_disp = np.rot90(rgba_grid_copy, k=k) if k != 0 else rgba_grid_copy
            # Dimensions after rotation
            if k in (1, -1, 3):  # 90° rotations
                disp_w = raw_h
                disp_h = raw_w
            else:
                disp_w = raw_w
                disp_h = raw_h
        else:
            # Fallback for backward compatibility
            rgba_grid_disp = np.rot90(rgba_grid_copy, k=-1) if rotate_clockwise else rgba_grid_copy
            disp_w = raw_h if rotate_clockwise else raw_w
            disp_h = raw_w if rotate_clockwise else raw_h
        extent = (0, disp_w, disp_h, 0)

        ax.imshow(
            rgba_grid_disp,
            interpolation="nearest",
            extent=extent,
            origin="upper",
            alpha=0.5,
        )

        # For grid mode, create legend from unique labels in the grid
        # Extract unique label indices from rgba_grid by matching colors
        mapped_labels = set()
        for lbl in label_order:
            color = hex_to_rgba(label_color_hex.get(lbl, "#ff00ff"))
            # Check if this color appears in the grid (approximate match)
            if np.any(np.all(np.abs(rgba_grid[:,:,:3] - color[:3]) < 0.01, axis=2)):
                # Exclude empty_label from legend if fill_empty_cell is False
                if fill_empty_cell or lbl != empty_label:
                    mapped_labels.add(lbl)

        legend_labels = [lbl for lbl in label_order[:5] if lbl in mapped_labels]
        for lbl in sorted(mapped_labels):
            if lbl not in legend_labels:
                legend_labels.append(lbl)

    else:
        # Annotation mode: plot circles from JSON
        ann_path = Path(row.get("annotation_path", ""))
        if not ann_path.exists():
            raise ValueError(f"annotation_path not found: {ann_path}")

        with open(ann_path, "r") as f:
            cells = json.load(f)

        if print_debug:
            print(f"\nAnnotation mode:")
            print(f"  - annotation file: {ann_path.name}")
            print(f"  - number of cells: {len(cells)}")
            if cells:
                x_coords = [c["center_x"] for c in cells]
                y_coords = [c["center_y"] for c in cells]
                print(f"  - X range in annotations: [{min(x_coords):.1f}, {max(x_coords):.1f}]")
                print(f"  - Y range in annotations: [{min(y_coords):.1f}, {max(y_coords):.1f}]")

        first_cell_printed = False
        for cell in cells:
            label = normalize_label(cell["label"], keep_labels, other_label)
            base_rgba = hex_to_rgba(label_color_hex.get(label, "#ff00ff"))

            if label == empty_label and not fill_empty_cell:
                face_rgba = with_alpha(base_rgba, 0.0)
            else:
                face_rgba = with_alpha(base_rgba, alpha_fill)

            x = cell["center_x"]
            y = cell["center_y"]

            # Transform annotation coordinates for display on rotated image
            if use_rotation_config and rot_cfg is not None:
                x_plot, y_plot = rot_cfg.transform_annotation_coords(x, y)
                transform_method = "rotation_config"
            elif rotate_clockwise:
                # Fallback for backward compatibility (assumes cw90)
                x_plot = h_orig - 1 - y
                y_plot = x
                transform_method = "fallback_cw90"
            else:
                x_plot = x
                y_plot = y
                transform_method = "none"

            if print_debug and not first_cell_printed:
                print(f"\nCoordinate transformation (first cell example):")
                print(f"  - transform method: {transform_method}")
                print(f"  - annotation coords: x={x:.1f}, y={y:.1f}")
                print(f"  - display coords:    x={x_plot:.1f}, y={y_plot:.1f}")
                if transform_method == "fallback_cw90":
                    print(f"  - formula: x_plot = h_orig - 1 - y = {h_orig} - 1 - {y:.1f} = {x_plot:.1f}")
                    print(f"            y_plot = x = {y_plot:.1f}")
                first_cell_printed = True

            circle = Circle(
                (x_plot, y_plot),
                cell["radius"],
                edgecolor=with_alpha(base_rgba, alpha_edge),
                facecolor=face_rgba,
                linewidth=1.0,
            )
            ax.add_patch(circle)

        mapped_labels = {normalize_label(c["label"], keep_labels, other_label) for c in cells}
        legend_labels = [lbl for lbl in label_order[:5] if lbl in mapped_labels]
        for lbl in sorted(mapped_labels):
            if lbl not in legend_labels:
                legend_labels.append(lbl)

    # Create legend
    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=hex_to_rgba(label_color_hex.get(lbl, "#ff00ff")),
            markeredgecolor="k",
            label=lbl,
            markersize=8,
        )
        for lbl in legend_labels
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=True)

    # Set title
    day_str = pd.to_datetime(row["day"]).strftime("%Y-%m-%d") if "day" in row else ""
    cam_str = f"cam {row['cam']}" if "cam" in row else ""
    title_parts = [p for p in [cam_str, day_str] if p]
    ax.set_title(" - ".join(title_parts) if title_parts else "Comb annotation")
    ax.axis("off")

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax 
