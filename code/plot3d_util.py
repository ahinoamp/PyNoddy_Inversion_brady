import vedo as vtkP
import time
import numpy as np
import meshio
import pandas as pd
import matplotlib.pylab as pl
from scipy.interpolate import griddata
import SimulationUtilities as sim
from glob import glob
from scipy import interpolate
import vedo as vtkP
from scipy.spatial import Delaunay


def getDXF_parsed_structure(output_name):
    filename = output_name + '.dxf'
#    doc = ezdxf.readfile(filename)
    cell_data = []
    xpoint = []
    ypoint = []
    zpoint = []
    with open(filename) as f:
        cntr=0
        faceCounter=0
        for line in f:
            if(cntr==(7+faceCounter*28)):
                cell_data.append(line)
                faceCounter=faceCounter+1
            elif(cntr==(9+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(11+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(13+(faceCounter-1)*28)):
                zpoint.append(float(line))

            elif(cntr==(15+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(17+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(19+(faceCounter-1)*28)):
                zpoint.append(float(line))

            elif(cntr==(21+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(23+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(25+(faceCounter-1)*28)):
                zpoint.append(float(line))

            cntr=cntr+1

    points = np.column_stack((np.asarray(xpoint, dtype=float),
                             np.asarray(ypoint, dtype=float),
                             np.asarray(zpoint, dtype=float)))
    cell_data.pop()
    cell_data = np.asarray(cell_data, dtype=object)
    print('Finished reading model')
   
#    Data = pd.DataFrame({'x': np.asarray(xpoint, dtype=float), 
#                         'y': np.asarray(ypoint, dtype=float),
#                         'z': np.asarray(zpoint, dtype=float),
#                         'cell_data': np.repeat(np.asarray(cell_data),3)})
    
#    Data.to_csv('xyz_strat.csv')
    return points, cell_data, faceCounter

def convertSurfaces2VTK(points, cell_data, faceCounter, outputOption = 1, fileprefix='Surface',  xy_origin=[0,0,0], num=0):
    
    # Choose output option
    num3Dfaces=faceCounter
    print('The number of triangle elements (cells/faces) is: ' + str(num3Dfaces))


    #apply origin transformation
    points[:, 0] = points[:, 0]+xy_origin[0]
    points[:, 1] = points[:, 1]+xy_origin[1]
    points[:, 2] = points[:, 2]+xy_origin[2]
    
    cell_data = pd.Series(cell_data.reshape((-1, )))

    CatCodes = np.zeros((len(cell_data),))
    filterB = (cell_data.str.contains('B')) 
    filterS = (cell_data.str.contains('S')) 

    CatCodes[filterB]= cell_data.loc[filterB].str[:-20].astype('category').cat.codes
    CatCodes[filterS]= -1*(cell_data.loc[filterS].str[:-12].astype('category').cat.codes+1)

    for i in range(1, len(CatCodes)):
        if(CatCodes[i]==0):
            CatCodes[i]=CatCodes[i-1]
            if(CatCodes[i-1]==0):
                CatCodes[i]=CatCodes[np.nonzero(CatCodes)[0][0]]

    UniqueCodes = np.unique(CatCodes)
    nSurfaces = len(UniqueCodes)

    Data = pd.DataFrame({'x': np.asarray(points[:, 0], dtype=float), 
                         'y': np.asarray(points[:, 1], dtype=float),
                         'z': np.asarray(points[:, 2], dtype=float),
                         'cell_data': np.repeat(np.asarray(CatCodes),3)})

    ## if you would like a single vtk file
    if (outputOption==2): 
        cells = np.zeros((num3Dfaces, 3),dtype ='int')
        i=0
        for f in range(num3Dfaces):
            cells[f,:]= [i, i+1, i+2]
            i=i+3
        meshio.write_points_cells(
            "Model.vtk",
            points,
            cells={'triangle':cells},
            cell_data= {'triangle': {'cat':CatCodes}}   
            )
    ## option 1: make a separate file for each surface
    else: 
        for i in range(nSurfaces):
            filterPoints = CatCodes==UniqueCodes[i]
            nCells = np.sum(filterPoints)
            Cells_i = np.zeros((nCells, 3),dtype ='int')
            cntr = 0
            for j in range(nCells):
                Cells_i[j]=[cntr, cntr+1, cntr+2]
                cntr=cntr+3
  
            meshio.write_points_cells(
                fileprefix+str(i)+".vtk",
                points[np.repeat(filterPoints,3), :],
                cells={'triangle':Cells_i}
                )
    
    return nSurfaces, points, CatCodes

def getxyz_sim_layer_top(P, output_name, layer_num = 4, dt_name='GT'):
    """Calculate the mismatch between observed and simulated granite top data."""

    get_model_dimensions(P)
        
    #load and reshape files
    filename = output_name+'.g12'
    LithologyCodes = np.genfromtxt(filename, delimiter='\t', dtype=int)
    LithologyCodes = LithologyCodes[:, 0:-1]

    lithology = np.zeros(P['shapeL'])
    for i in range(P['shapeL'][2]):
        startIdx = P['shapeL'][1]*i 
        stopIdx = P['shapeL'][1]*(i+1)
        lithology[:,:,i] = LithologyCodes[startIdx:stopIdx,:].T
    lithology = lithology[::-1,:,:]

    # Find the first indices of the top of granite (in the z direction)
    topgraniteIdx = np.argmax(lithology==layer_num, axis=2) 
    topgranite = P['zmax']-topgraniteIdx*float(P['cubesize'])
  
#    if(np.sum(topgranite<-1500)):
#        print('Top of granite is very high in ' + str(np.sum(topgranite<-1500)) +' spots')
        
    P['xLith'] = np.linspace(P['xminL'], P['xmaxL'], P['nxL'], dtype=np.float32)+P['xmin']
    P['yLith'] = np.linspace(P['yminL'], P['ymaxL'], P['nyL'], dtype=np.float32)+P['ymin']
    P['yyLith'], P['xxLith'] = np.meshgrid(P['yLith'], P['xLith'], indexing='ij')
    #get only the valid values
    filteroutNan = ~np.isnan(topgranite) 
    x1 = P['xxLith'][filteroutNan]
    y1 = P['yyLith'][filteroutNan]
    newtopgranite = topgranite[filteroutNan]
   
    
    GT_Sim = interpolate.griddata((x1, y1), newtopgranite.ravel(),
                              (P[dt_name]['xObs'], P[dt_name]['yObs']), method='linear')
    
    return GT_Sim
    
def get_model_dimensions(P):
    """Load information about model discretisation from .g00 file"""

    output_name = P['output_name']
    filename = output_name+'.g00'

    filelines = open(filename).readlines() 
    for line in filelines:
        if 'NUMBER OF LAYERS' in line:
            P['nzL'] = int(line.split("=")[1])
        elif 'LAYER 1 DIMENSIONS' in line:
            (P['nxL'], P['nyL']) = [int(l) for l in line.split("=")[1].split(" ")[1:]]
        elif 'UPPER SW CORNER' in line:
            l = [float(l) for l in line.split("=")[1].split(" ")[1:]]
            (P['xminL'], P['yminL'], P['zmaxL']) = l
        elif 'LOWER NE CORNER' in line:
            l = [float(l) for l in line.split("=")[1].split(" ")[1:]]
            (P['xmaxL'], P['ymaxL'], P['zminL']) = l
        elif 'NUM ROCK' in line:
            n_rocktypes = int(line.split('=')[1])

    P['shapeL'] = (P['nyL'], P['nxL'], P['nzL'])

def CalculatePlotStructure(modelfile, plot, cubesize = 250,  
                           xy_origin=[317883,4379246, 1200-4000], plotwells =1,
                           outputOption = 1, outputfolder = '', num=0, Windows=False):
    
    output_name = 'PostProcessing/3dmodel'
    outputoption = 'ALL'

    #Calculate the model
    start = time.time()
    sim.calculate_model(modelfile, output_name, outputoption, Windows=Windows)
    sim.calculate_model(modelfile, output_name, 'TOPOLOGY', Windows=Windows)
    end = time.time()
    print('Calculation time took '+str(end - start) + ' seconds')

    ## Now need to change the DXF file (mesh format) to VTK. 
    ## This is slow unfortunately and I'm sure can be optimized
    start = time.time()
    points, cell_data, faceCounter = getDXF_parsed_structure(output_name)
    end = time.time()
    print('Parsing time took '+str(end - start) + ' seconds')


    ## Make a vtk file for each surface (option 1) 
    # or make a single vtk file for all surfaces (option 2)
    fileprefix = outputfolder+'Surface'
    start = time.time()
    nSurfaces, points, CatCodes = convertSurfaces2VTK(points, cell_data,
                                                      faceCounter, outputOption,
                                                      fileprefix, num=num, 
                                                      xy_origin=xy_origin)   
    end = time.time()
    print('Convert 2 VTK time took '+str(end - start) + ' seconds')

#     ## Now get the lithology data
#     N1 = pynoddy.output.NoddyOutput(output_name)
#     Lithology = N1.block
# #    Lithology=np.swapaxes(Lithology,0,2)


#     lithology = N1.block

#     [maxX, maxY, maxZ] = np.max(points, axis=0)
#     [minX, minY, minZ] = np.min(points, axis=0)
#     minZ = xy_origin[2]
#     x = np.linspace(minX, maxX, N1.nx, dtype=np.float32)
#     y = np.linspace(minY, maxY, N1.ny, dtype=np.float32)
#     z = np.linspace(xy_origin[2], maxZ, N1.nz, dtype=np.float32)
# #    z = np.linspace(0, 4000, N1.nz, dtype=np.float32)

#     delx = x[1]-x[0]
#     dely = y[1]-y[0]
#     delz = z[1]-z[0]
    
#     xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

#     CoordXYZ = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1),zz.reshape(-1,1)), axis=1)

#     Lithology2 = griddata(CoordXYZ, np.transpose(lithology, axes =(2, 1, 0)).reshape(-1,), (xx, yy, zz), method='nearest')


#     vol = vtkP.Volume(Lithology2, c='jet', spacing=[N1.delx, N1.dely,N1.delz], origin =[xy_origin[0]+N1.xmin, xy_origin[1]+N1.ymin, xy_origin[2]+N1.zmin])
#     lego = vol.legosurface(-1, np.max(Lithology)*2).opacity(0.95).c('jet')
#     plot += lego

    colors = pl.cm.jet(np.linspace(0,1,nSurfaces))

    for i in range(nSurfaces):
        filename = fileprefix+str(i)+'.vtk'
        e=vtkP.load(filename).tomesh(fill=True).c(colors[i, 0:3])
        plot += e

    plot_basement_top = True

    if(plot_basement_top):
    #Granite top observations
        P={}
        P['cubesize'] = cubesize
        P['output_name'] = output_name
        P['xy_origin'] = xy_origin
        P['GT'] = {}
        GraniteTopObs = pd.read_csv('Data/BradyTopBasement.csv')
        P['GT']['xObs'] = GraniteTopObs['x'].values
        P['GT']['yObs'] = GraniteTopObs['y'].values
        P['GT']['Obs'] = GraniteTopObs['z'].values
        P['GT']['nObsPoints'] = len(P['GT']['Obs'])
    
        P['MVT'] = {}
        GraniteTopObs = pd.read_csv('Data/BradyTopMioceneVolc.csv')
        P['MVT']['xObs'] = GraniteTopObs['x'].values
        P['MVT']['yObs'] = GraniteTopObs['y'].values
        P['MVT']['Obs'] = GraniteTopObs['z'].values
        P['MVT']['nObsPoints'] = len(P['MVT']['Obs'])
        P['zmax'] = 1200
        P['xmin'] = P['xy_origin'][0]
    
        P['ymin'] = P['xy_origin'][1]
    
        P['zmin'] = P['xy_origin'][2]
        dt_name='GT'
        GT_Sim_GT = getxyz_sim_layer_top(P, output_name, layer_num = 4, dt_name=dt_name)
        GT_pts = np.concatenate((P[dt_name]['xObs'].reshape((-1,1)), 
                                 P[dt_name]['yObs'].reshape((-1,1)),
                                 GT_Sim_GT.reshape((-1,1))), axis=1)
        gt_sim_viz = vtkP.Points(GT_pts, c="g", r=60)
        gt_sim_viz.name = 'simulated granite top'
        plot += gt_sim_viz.flag()

        GT_pts_obs = np.concatenate((P[dt_name]['xObs'].reshape((-1,1)), 
                                 P[dt_name]['yObs'].reshape((-1,1)),
                                 P[dt_name]['Obs'].reshape((-1,1))), axis=1)
        gt_obs_viz = vtkP.Points(GT_pts_obs, c="k", r=60)
        gt_obs_viz.name = 'observed granite top'
        plot += gt_obs_viz.flag()

        dt_name='MVT'
        GT_Sim_MVT = getxyz_sim_layer_top(P, output_name, layer_num = 2, dt_name=dt_name)
        MVT_pts = np.concatenate((P[dt_name]['xObs'].reshape((-1,1)), 
                                  P[dt_name]['yObs'].reshape((-1,1)), 
                                  GT_Sim_MVT.reshape((-1,1))), axis=1)
        #P[dt_name]['xObs'], P[dt_name]['yObs']        
        mvt_sim_viz = vtkP.Points(MVT_pts, c="g", r=60)
        mvt_sim_viz.name = 'simulated miocene volcanic top'
        plot += mvt_sim_viz.flag()
        
        MVT_pts_obs = np.concatenate((P[dt_name]['xObs'].reshape((-1,1)), 
                                 P[dt_name]['yObs'].reshape((-1,1)),
                                 P[dt_name]['Obs'].reshape((-1,1))), axis=1)
        mvt_obs_viz = vtkP.Points(MVT_pts_obs, c="k", r=60)
        mvt_obs_viz.name = 'observed miocene volcanic top'
        plot += mvt_obs_viz.flag()      
        
        # fault markers
        P['DataTypes'] = ['FaultMarkers']
        P['FaultMarkers'] = {}
        FaultMarkers= pd.read_csv('Data/BradyWellsFaults.csv')
        FaultMarkers['wellid'] = FaultMarkers.groupby(['WellName']).ngroup()
        P['FaultMarkers']['Obs'] = FaultMarkers
        P['HypP']={}
        P['iterationNum']=0
        P['HypP']['MaxFaultMarkerError'] = 550
        WellPathsOrig = pd.read_csv('Data/AllBrady3DWells.csv')
        Wellnames = WellPathsOrig['WellName']
        FaultMarkerWells = np.unique(P['FaultMarkers']['Obs']['WellName'])
        filterWells = np.isin(Wellnames, FaultMarkerWells)
        WellPathsTracers = WellPathsOrig[filterWells]
        nWells=len(FaultMarkerWells)
        zWells = np.zeros((nWells,2))
        idWells = np.zeros((nWells,2))
        topXWell = np.zeros((nWells,))
        topYWell = np.zeros((nWells,))
        idPlotWell = np.zeros((nWells,), dtype=int)
    
        WellsAtMaxZ = WellPathsTracers[WellPathsTracers['Zm'] == WellPathsTracers.groupby('WellName')['Zm'].transform('max')]
    
        for i in range(nWells):
            filterWell = WellPathsTracers['WellName']==FaultMarkerWells[i]
            z = WellPathsTracers.loc[filterWell, 'Zm']
            zWells[i, 0] = np.min(z)
            zWells[i, 1] = np.max(z)
            idWells[i,:] = i
     
            filterMaxTable = WellsAtMaxZ['WellName']==FaultMarkerWells[i]
            topXWell[i] = WellsAtMaxZ.loc[filterMaxTable, 'Xm'].values[0]
            topYWell[i] = WellsAtMaxZ.loc[filterMaxTable, 'Ym'].values[0]
            idPlotWell[i] = i
            
        P['FaultMarkers']['WellData'] =  WellPathsTracers[WellPathsTracers['Zm']<1200].copy(deep=True)
        P['FaultMarkers']['WellData']['id'] = P['FaultMarkers']['WellData'].groupby(['WellName']).ngroup()
    
        P['zWells'] = zWells.T
        P['idWells'] = idWells.T
        P['topXWell'] = topXWell.T
        P['topYWell'] = topYWell.T
        P['idPlotWell'] = idPlotWell.T
    
        P['FaultMarkers']['nObsPoints'] = len(P['FaultMarkers']['Obs'])
        P['FaultMarkers']['xObs'] = P['FaultMarkers']['Obs']['X']
        P['FaultMarkers']['yObs'] = P['FaultMarkers']['Obs']['Y']
        sim.calc_fault_markers(P)
        fault_markers = np.concatenate((np.asarray(P['FaultMarkers']['simX']).reshape((-1,1)), 
                              np.asarray(P['FaultMarkers']['simY']).reshape((-1,1)), 
                              np.asarray(P['FaultMarkers']['simZ']).reshape((-1,1))), axis=1)
        fault_markers_viz = vtkP.Points(fault_markers, c="o", r=60)
        fault_markers_viz.name = 'simulated fault markers'
        plot += fault_markers_viz.flag()
        
    return points

def plot_3d_model(modelfile, cubesize, plot, xy_origin = [316448, 4379166, -2700], Windows=False):

    points = CalculatePlotStructure(modelfile, plot, cubesize = cubesize, xy_origin=xy_origin, Windows=Windows)
    
if __name__== "__main__":

    
    best_model_file = 'try2.his'
    xy_origin = [325233.059, 4404112, -2700]
    xy_extent = [4950, 6150, 3900]

        
    vtkP.settings.embedWindow('k3d') #you can also choose to change to itkwidgets, k3d
    
    cubesize = 100
    if(cubesize==100):
        xy_extent = [5000,	6200, 3900]
    else:
        xy_extent = [4950,	6150, 3900]
        
    plot = vtkP.Plotter(axes=1, bg='white', interactive=1)
    plot_3d_model(best_model_file, cubesize, plot, xy_origin, Windows=True)

    # add topography
    ##################
    # perform a 2D Delaunay triangulation to get the cells from the point cloud
    landSurfacePD = pd.read_csv("Data/BradysDEM.csv")
    filterLimits =  ((landSurfacePD['x']>xy_origin[0]) & 
                     (landSurfacePD['y']>xy_origin[1])   
                     & (landSurfacePD['x']<(xy_origin[0]+xy_extent[0])) 
                     & (landSurfacePD['y']<(xy_origin[1]+xy_extent[1]))) 
    landSurfacePD = landSurfacePD[filterLimits]
    landSurfacePD = landSurfacePD[['x', 'y', 'z']].values
    tri = Delaunay(landSurfacePD[:, 0:2])
    
    # create a mesh object for the land surface
    landSurface = vtkP.Mesh([landSurfacePD, tri.simplices])
    
    # in order to color it by the elevation, we use the z values of the mesh
    zvals = landSurface.points()[:, 2]
    landSurface.pointColors(zvals, cmap="terrain", vmin=1000)
    landSurface.name = "Land Surface" # give the object a name
    
    plot+=landSurface

    # add fault markers
    ###########################
    ObsFaultMarkers = pd.read_csv('Data/BradyWellsFaults.csv')
    fault_markers = np.concatenate((ObsFaultMarkers['X'].values.reshape((-1,1)), 
                              ObsFaultMarkers['Y'].values.reshape((-1,1)), 
                              ObsFaultMarkers['Z'].values.reshape((-1,1))), axis=1)
    #P[dt_name]['xObs'], P[dt_name]['yObs']        
    fault_markers_viz = vtkP.Points(fault_markers, c="b", r=60)
    fault_markers_viz.name = 'observed fault markers'
    plot += fault_markers_viz.flag()
        
    # add wellbores
    ###########################
    wells = pd.read_csv('Data/AllBrady3DWells.csv')
    well_names = pd.unique(wells['WellName'])
    for wn in well_names:
        filterWN = wells['WellName']==wn
        well = wells[filterWN]
        well=well[['Xm', 'Ym', 'Zm']].values
        Well = vtkP.Line(well).color('red').lw(2)
        Well.name = wn
        plot += Well.flag()

    plot.show(viewup='z')
    
