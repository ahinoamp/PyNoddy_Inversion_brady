# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:13:20 2020

@author: ahinoamp
"""

class _HisTemplates():
    header = """#Filename = simple_two_faults.his
#Date Saved = 24/3/2014 14:21:0
FileType = 111
Version = 7.11"""

    strati_layer = """    Unit Name    = $NAME$
    Height    = $Height$
    Apply Alterations    = ON
    Density    = $Density$
    Anisotropic Field    = 0
    MagSusX    = $MagSus$
    MagSusY    = $MagSus$
    MagSusZ    = $MagSus$
    MagSus Dip    = 9.00e+001
    MagSus DipDir    = 9.00e+001
    MagSus Pitch    = 0.00e+000
    Remanent Magnetization    = 0
    Inclination    =  30.00
    Angle with the Magn. North    =  30.00
    Strength    = 1.60e-003
    Color Name    = Color 92
    Red    = $RED$
    Green    = $GREEN$
    Blue    = $BLUE$ """

    fault_start = """    Geometry    = Curved
    Movement    = Hanging Wall
    X    = $X$
    Y    = $Y$
    Z    =   $Z$
    Dip Direction    =  $Dip Direction$
    Dip    =  $Dip$
    Pitch    =  $Pitch$
    Slip    = $Slip$
    Rotation    = 30
    Amplitude    = $Amplitude$
    Radius    = 1000
    XAxis    = $XAxis$
    YAxis    = $YAxis$
    ZAxis    = $ZAxis$
    Cyl Idx    =   0.00
    Profile Pitch    = $Profile Pitch$
    Color Name    = Custom Colour 8
    Red    = 0
    Green    = 0
    Blue    = 254
    Fourier Series
        Term A 0    =   0.00
        Term B 0    =   0.00
        Term A 1    =   0.00
        Term B 1    =   1.00
        Term A 2    =   0.00
        Term B 2    =   0.00
        Term A 3    =   0.00
        Term B 3    =   0.00
        Term A 4    =   0.00
        Term B 4    =   0.00
        Term A 5    =   0.00
        Term B 5    =   0.00
        Term A 6    =   0.00
        Term B 6    =   0.00
        Term A 7    =   0.00
        Term B 7    =   0.00
        Term A 8    =   0.00
        Term B 8    =   0.00
        Term A 9    =   0.00
        Term B 9    =   0.00
        Term A 10    =   0.00
        Term B 10    =   0.00
    Name    = Fault Plane
    Type    = 1
    Join Type     = $Join Type$
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 6.280000
    Min Y Scale    = -1.000000
    Max Y Scale    = 1.000000
    Scale Origin    = 0.000000
    Min Y Replace    = -1.000000
    Max Y Replace    = 1.000000"""
    
    fault_end = """    Alteration Type     = NONE
    Num Profiles    = 12
    Name    = Density
    Type    = 2
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = 0.000000
    Max Y Scale    = 4.000000
    Scale Origin    = 1.000000
    Min Y Replace    = 0.000000
    Max Y Replace    = 10.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = -50
        Point X    = 628
        Point Y    = -50
    Name    = Anisotropy
    Type    = 3
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -10.000000
    Max Y Scale    = 10.000000
    Scale Origin    = 0.000000
    Min Y Replace    = -10.000000
    Max Y Replace    = 10.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = - X Axis (Sus)
    Type    = 4
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -5.000000
    Max Y Scale    = 5.000000
    Scale Origin    = 0.000000
    Min Y Replace    = 2.000000
    Max Y Replace    = 8.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = - Y Axis (Sus)
    Type    = 5
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -5.000000
    Max Y Scale    = 5.000000
    Scale Origin    = 0.000000
    Min Y Replace    = 2.000000
    Max Y Replace    = 8.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = - Z Axis (Sus)
    Type    = 6
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -5.000000
    Max Y Scale    = 5.000000
    Scale Origin    = 0.000000
    Min Y Replace    = 2.000000
    Max Y Replace    = 8.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = - Dip (Sus)
    Type    = 7
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -180.000000
    Max Y Scale    = 180.000000
    Scale Origin    = 1.000000
    Min Y Replace    = -180.000000
    Max Y Replace    = 180.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 1
        Point X    = 628
        Point Y    = 1
    Name    = - Dip Dir (Sus)
    Type    = 8
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -360.000000
    Max Y Scale    = 360.000000
    Scale Origin    = 1.000000
    Min Y Replace    = -360.000000
    Max Y Replace    = 360.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = - Pitch (Sus)
    Type    = 9
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -360.000000
    Max Y Scale    = 360.000000
    Scale Origin    = 1.000000
    Min Y Replace    = -360.000000
    Max Y Replace    = 360.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = Remanence
    Type    = 10
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -10.000000
    Max Y Scale    = 10.000000
    Scale Origin    = 0.000000
    Min Y Replace    = -10.000000
    Max Y Replace    = 10.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = - Declination (Rem)
    Type    = 11
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -360.000000
    Max Y Scale    = 360.000000
    Scale Origin    = 1.000000
    Min Y Replace    = -360.000000
    Max Y Replace    = 360.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = - Inclination (Rem)
    Type    = 12
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -360.000000
    Max Y Scale    = 360.000000
    Scale Origin    = 1.000000
    Min Y Replace    = -360.000000
    Max Y Replace    = 360.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Name    = - Intensity (Rem)
    Type    = 13
    Join Type     = LINES
    Graph Length    = 200.000000
    Min X    = 0.000000
    Max X    = 0.000000
    Min Y Scale    = -5.000000
    Max Y Scale    = 5.000000
    Scale Origin    = 0.000000
    Min Y Replace    = -5.000000
    Max Y Replace    = 5.000000
    Num Points    = 2
        Point X    = 0
        Point Y    = 0
        Point X    = 628
        Point Y    = 0
    Surface Type    = FLAT_SURFACE
    Surface Filename    =      
    Surface Directory    = \\psf\Home
    Surface XDim    = 0.000000
    Surface YDim    = 0.000000
    Surface ZDim    = 0.000000
    Name    = $NAME$"""

    # AK 2014-10
    tilt = """    X    =   4500
    Y    =   4500
    Z    =   2000
    Rotation     =  $Tilt_Rotation$
    Plunge Direction     = $Tilt_Plunge Direction$
    Plunge     =   0
    Name    = Tilt"""

    plug = """    Type	= Ellipsoidal
	Merge Events	= 0
	X	= $Plug_X$
	Y	= $Plug_Y$
	Z	= $Plug_Z$
	Dip Direction	=  $Plug_Dip Direction$
	Dip	=  90.00
	Pitch	 =   0.00
	Radius	 = $Plug_Radius$
	ApicalAngle	 =  30.00
	B-value	 = 2000.00
	A-value	 = $Plug_ZAxis$
	B-value	 = $Plug_YAxis$
	C-value	 = $Plug_XAxis$
	Alteration Type 	= NONE
	Num Profiles	= 12
	Name	= Density
	Type	= 2
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= 0.000000
	Max Y Scale	= 4.000000
	Scale Origin	= 1.000000
	Min Y Replace	= 0.000000
	Max Y Replace	= 10.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= -50
		Point X	= 628
		Point Y	= -50
	Name	= Anisotropy
	Type	= 3
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -10.000000
	Max Y Scale	= 10.000000
	Scale Origin	= 0.000000
	Min Y Replace	= -10.000000
	Max Y Replace	= 10.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= - X Axis (Sus)
	Type	= 4
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -5.000000
	Max Y Scale	= 5.000000
	Scale Origin	= 0.000000
	Min Y Replace	= 2.000000
	Max Y Replace	= 8.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= - Y Axis (Sus)
	Type	= 5
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -5.000000
	Max Y Scale	= 5.000000
	Scale Origin	= 0.000000
	Min Y Replace	= 2.000000
	Max Y Replace	= 8.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= - Z Axis (Sus)
	Type	= 6
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -5.000000
	Max Y Scale	= 5.000000
	Scale Origin	= 0.000000
	Min Y Replace	= 2.000000
	Max Y Replace	= 8.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= - Dip (Sus)
	Type	= 7
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -180.000000
	Max Y Scale	= 180.000000
	Scale Origin	= 1.000000
	Min Y Replace	= -180.000000
	Max Y Replace	= 180.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 1
		Point X	= 628
		Point Y	= 1
	Name	= - Dip Dir (Sus)
	Type	= 8
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -360.000000
	Max Y Scale	= 360.000000
	Scale Origin	= 1.000000
	Min Y Replace	= -360.000000
	Max Y Replace	= 360.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= - Pitch (Sus)
	Type	= 9
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -360.000000
	Max Y Scale	= 360.000000
	Scale Origin	= 1.000000
	Min Y Replace	= -360.000000
	Max Y Replace	= 360.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= Remanence
	Type	= 10
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -10.000000
	Max Y Scale	= 10.000000
	Scale Origin	= 0.000000
	Min Y Replace	= -10.000000
	Max Y Replace	= 10.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= - Declination (Rem)
	Type	= 11
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -360.000000
	Max Y Scale	= 360.000000
	Scale Origin	= 1.000000
	Min Y Replace	= -360.000000
	Max Y Replace	= 360.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= - Inclination (Rem)
	Type	= 12
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -360.000000
	Max Y Scale	= 360.000000
	Scale Origin	= 1.000000
	Min Y Replace	= -360.000000
	Max Y Replace	= 360.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Name	= - Intensity (Rem)
	Type	= 13
	Join Type 	= LINES
	Graph Length	= 200.000000
	Min X	= 0.000000
	Max X	= 0.000000
	Min Y Scale	= -5.000000
	Max Y Scale	= 5.000000
	Scale Origin	= 0.000000
	Min Y Replace	= -5.000000
	Max Y Replace	= 5.000000
	Num Points	= 2
		Point X	= 0
		Point Y	= 0
		Point X	= 628
		Point Y	= 0
	Unit Name	= Gabbro
	Height	= 5500
	Apply Alterations	= ON
	Density	= $Plug_Density$
	Anisotropic Field	= 0
	MagSusX	= $Plug_MagSus$
	MagSusY	= $Plug_MagSus$
	MagSusZ	= $Plug_MagSus$
	MagSus Dip	= 9.00e+001
	MagSus DipDir	= 9.00e+001
	MagSus Pitch	= 0.00e+000
	Remanent Magnetization	= 0
	Inclination	=  30.00
	Angle with the Magn. North	=  30.00
	Strength	= 1.00e-002
	Color Name	= Color 28
	Red	= 216
	Green	= 255
	Blue	= 0
	Name	= Plug"""
    # everything below events
    footer = """
#BlockOptions
    Number of Views    = 1
    Current View    = 0
    NAME    = Default
    Origin X    =   0.00
    Origin Y    =   0.00
    Origin Z    = $origin_z$
    Length X    = $extent_x$
    Length Y    = $extent_y$
    Length Z    = $extent_z$
    Geology Cube Size    =  $cube_size$
    Geophysics Cube Size    = $cube_size$

#GeologyOptions
    Scale    =  10.00
    SectionDec    =  90.00
    WellDepth    = 5000.00
    WellAngleZ    =   0.00
    BoreholeX    =   0.00
    BoreholeX    =   0.00
    BoreholeX    = 5000.00
    BoreholeDecl    =  90.00
    BoreholeDip    =   0.00
    BoreholeLength    = 5000.00
    SectionX    =   0.00
    SectionY    =   0.00
    SectionZ    = 5000.00
    SectionDecl    =  90.00
    SectionLength    = 10000.00
    SectionHeight    = 5000.00
    topofile    = FALSE
    Topo Filename    =    
    Topo Directory    = .
    Topo Scale    =   1.00
    Topo Offset    =   0.00
    Topo First Contour    = 100.00
    Topo Contour Interval    = 100.00
    Chair Diagram    = FALSE
    Chair_X    = 5000.00
    Chair_Y    = 3500.00
    Chair_Z    = 2500.00

#GeophysicsOptions
    GPSRange     = 1200
	Declination	=  13.20
	Inclination	=  63.40
	Intensity	= 49605.00
    Field Type    = FIXED
    Field xPos    =   0.00
    Field yPos    =   0.00
    Field zPos    = 5000.00
    Inclination Ori    =   0.00
    Inclination Change    =   0.00
    Intensity Ori    =  90.00
    Intensity Change    =   0.00
    Declination Ori    =   0.00
    Declination Change    =   0.00
    Altitude    =  80.00
    Airborne=     FALSE
    Calculation Method    = SPATIAL
    Spectral Padding Type    = RECLECTION_PADDING
    Spectral Fence    = 100
    Spectral Percent    = 100
    Constant Boxing Depth    =   0.00
    Clever Boxing Ratio    =   1.00
    Deformable Remanence=     FALSE
    Deformable Anisotropy=     TRUE
    Vector Components=     FALSE
    Project Vectors=     TRUE
    Pad With Real Geology=     FALSE
    Draped Survey=     FALSE

#3DOptions
    Declination    = 150.000000
    Elevation    = 30.000000
    Scale    = 1.000000
    Offset X    = 1.000000
    Offset Y    = 1.000000
    Offset Z    = 1.000000
    Fill Type    = 2

#ProjectOptions
    Susceptibility Units    = SI
    Geophysical Calculation    = 2
    Calculation Type    = LOCAL_JOB
    Length Scale    = 0
    Printing Scale    = 1.000000
    Image Scale    = 10.000000
    New Windows    = FALSE
    Background Red Component    = 254
    Background Green Component    = 254
    Background Blue Component    = 254
    Internet Address    = 255.255.255.255
    Account Name    =       
    Noddy Path    = ./noddy
    Help Path    = iexplore %h
    Movie Frames Per Event    = 3
    Movie Play Speed    =  10.00
    Movie Type    = 0
    Gravity Clipping Type    = RELATIVE_CLIPPING
    Gravity Image Display Clip Min    = 0.000000
    Gravity Image Display Clip Max    = 100.000000
    Gravity Image Display Type    = GREY
    Gravity Image Display Num Contour    = 25
    Magnetics Clipping Type    = RELATIVE_CLIPPING
    Magnetics Image Display Clip Min    = 0.000000
    Magnetics Image Display Clip Max    = 100.000000
    Magnetics Image Display Type    = GREY
    Magnetics Image Display Num Contour    = 25
    False Easting    = 0.000000
    False Northing    = 0.000000

#Window Positions
    Num Windows    = 16
    Name    = Block Diagram
    X    = 60
    Y    = 60
    Width    = 500
    Height    = 300
    Name    = Movie
    X    = 60
    Y    = 60
    Width    = -1
    Height    = -1
    Name    = Well Log
    X    = 60
    Y    = 60
    Width    = 400
    Height    = 430
    Name    = Section
    X    = 14
    Y    = 16
    Width    = 490
    Height    = -1
    Name    = Topography Map
    X    = 60
    Y    = 60
    Width    = 490
    Height    = 375
    Name    = 3D Topography Map
    X    = 60
    Y    = 60
    Width    = 490
    Height    = 375
    Name    = 3D Stratigraphy
    X    = 60
    Y    = 60
    Width    = 490
    Height    = 375
    Name    = Line Map
    X    = 60
    Y    = 60
    Width    = 490
    Height    = -1
    Name    = Profile - From Image
    X    = 60
    Y    = 60
    Width    = 490
    Height    = 600
    Name    = Sterographic Projections
    X    = 60
    Y    = 60
    Width    = 430
    Height    = 430
    Name    = Stratigraphic Column
    X    = 60
    Y    = 60
    Width    = 230
    Height    = 400
    Name    = Image
    X    = 30
    Y    = 30
    Width    = -1
    Height    = -1
    Name    = Contour
    X    = 30
    Y    = 30
    Width    = -1
    Height    = -1
    Name    = Toolbar
    X    = 10
    Y    = 0
    Width    = -1
    Height    = -1
    Name    = His
    X    = 229
    Y    = 160
    Width    = 762
    Height    = 898
    Name    = His
    X    = 229
    Y    = 160
    Width    = 762
    Height    = 898

#Icon Positions
    Num Icons    = 3
    Row    = 1
    Column    = 1
    X Position    = 1
    Y Position    = 1
    Row    = 1
    Column    = 2
    X Position    = 4
    Y Position    = 1
    Row    = 1
    Column    = 3
    X Position    = 7
    Y Position    = 1
    Floating Menu Rows    = 1
    Floating Menu Cols    = 24
End of Status Report"""