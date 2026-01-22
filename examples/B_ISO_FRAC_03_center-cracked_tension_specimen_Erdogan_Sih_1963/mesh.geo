//// Parameters
// Options
Mesh.Algorithm = 5;
// Geometry
L    = DefineNumber[ 457.2e-3, Name "Parameters/L"    ];
W    = DefineNumber[ 228.6e-3, Name "Parameters/W"    ];
a    = DefineNumber[  25.4e-3, Name "Parameters/a"    ];
beta = DefineNumber[       60, Name "Parameters/beta" ];
alpha = Pi/2-beta*Pi/180;
// Spatial step
h = DefineNumber[ 4e-4, Name "Parameters/h"];
hmax = 25*h;

//// Points
// Bot
Point(11) = {-W/2, -L/2, 0, hmax};
Point(12) = {+W/2, -L/2, 0, hmax};
// Top
Point(21) = {-W/2, +L/2, 0, hmax};
Point(22) = {+W/2, +L/2, 0, hmax};
// Crack
Point(31) = {+a*Cos(alpha) + h/2*Sin(alpha), +a*Sin(alpha) - h/2*Cos(alpha), 0, hmax};
Point(32) = {+a*Cos(alpha) - h/2*Sin(alpha), +a*Sin(alpha) + h/2*Cos(alpha), 0, hmax};
Point(33) = {-a*Cos(alpha) - h/2*Sin(alpha), -a*Sin(alpha) + h/2*Cos(alpha), 0, hmax};
Point(34) = {-a*Cos(alpha) + h/2*Sin(alpha), -a*Sin(alpha) - h/2*Cos(alpha), 0, hmax};

//// Lines
// Outer boundary
Line(11) = {11, 12};
Line(12) = {12, 22};
Line(13) = {22, 21};
Line(14) = {21, 11};
// Crack
Line(21) = {31, 32};
Line(22) = {32, 33};
Line(23) = {33, 34};
Line(24) = {34, 31};

//// Surfaces
Curve Loop(1) = {11, 12, 13, 14};
Curve Loop(2) = {21, 22, 23, 24};
Plane Surface(1) = {1, 2};

//// Physical groups
// Domain
Physical Surface("domain", 11) = {1};
// Boundaries
Physical Curve("bot", 9) = {11};
Physical Curve("top", 10) = {13};
// Crack
Physical Curve("crack", 12) = {21, 22, 23, 24};

////
// Element size
/////////////////
// Create a lines for mesh refinement
Point(101) = {+W/2, +a/2 * Sin(alpha), 0, hmax};
Point(102) = {-W/2, -a/2 * Sin(alpha), 0, hmax};
Line(101) = {101, 31};
Line(102) = {102, 33};
// Create a distance field
Field[1] = Distance;
Field[1].CurvesList = {21, 22, 23, 24, 101, 102};
Field[1].Sampling = 100;
// Use a distance field and a threshold to set the element size
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = 20 * h;
Field[2].DistMax = 50 * h;
Field[2].SizeMin = h;
Field[2].SizeMax = hmax;
// Set the treshold field as the background field
Background Field = 2;

