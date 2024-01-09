////
// Parameters
///////////////
// Geometry
L    = DefineNumber[ 457.2e-3, Name "Parameters/L"    ];
W    = DefineNumber[ 228.6e-3, Name "Parameters/W"    ];
a    = DefineNumber[  25.4e-3, Name "Parameters/a"    ];
aw   = DefineNumber[    1e-12, Name "Parameters/aw"    ];
beta = DefineNumber[       60, Name "Parameters/beta" ];
// Spatial step
hc   = DefineNumber[ 0.2e-3, Name "Parameters/hc"]; // Crack Real value: 0.1e-3
hb   = DefineNumber[  12*hc, Name "Parameters/hb"]; // Boundary

////
// Points
///////////
// Boundaries
Point(1) = {-W/2, -L/2, 0, hb};
Point(2) = {+W/2, -L/2, 0, hb};
Point(3) = {+W/2, +L/2, 0, hb};
Point(4) = {-W/2, +L/2, 0, hb};
// Crack
Point(5) = {-a*Cos(Pi/2-beta*Pi/180), -a*Sin(Pi/2-beta*Pi/180)-aw, 0, hc};
Point(6) = {+a*Cos(Pi/2-beta*Pi/180), +a*Sin(Pi/2-beta*Pi/180)-aw, 0, hc};
Point(7) = {+a*Cos(Pi/2-beta*Pi/180), +a*Sin(Pi/2-beta*Pi/180)+aw, 0, hc};
Point(8) = {-a*Cos(Pi/2-beta*Pi/180), -a*Sin(Pi/2-beta*Pi/180)+aw, 0, hc};

////
// Lines
//////////
// Boundaries
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
// Crack
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

////
// Surfaces
/////////////
// Boundaries
Curve Loop(1) = {1, 2, 3, 4};
// Crack
Curve Loop(2) = {5, 6, 7, 8};
// Surface
Plane Surface(1) = {1, 2};

////
// Physical groups
////////////////////
// Domain
Physical Surface("domain", 11) = {1};
// Boundaries
Physical Curve("bot", 9) = {1};
Physical Curve("top", 10) = {3};
// Crack
Physical Curve("crack", 12) = {5,6,7,8};

////
// Element size
/////////////////
// Define a line near the crack
Point(9)  = {-W/2, 0, 0, hc};
Point(10) = { W/2, 0, 0, hc};
Line(9) = {9, 10};
// Create a distance field
Field[1] = Distance;
Field[1].CurvesList = {9};
Field[1].Sampling = 10000;
// Use a distance field and a threshold to set the element size
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = 1.5*a*Sin(Pi/2-beta*Pi/180);
Field[2].DistMax = 2.0*a*Sin(Pi/2-beta*Pi/180);
Field[2].SizeMin = hc;
Field[2].SizeMax = hb;
// Set the treshold field as the background field
Background Field = 2;

