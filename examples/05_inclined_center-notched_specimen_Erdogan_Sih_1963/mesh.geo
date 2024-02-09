//// Parameters
// Geometry
L    = DefineNumber[ 457.2e-3, Name "Parameters/L"    ];
W    = DefineNumber[ 228.6e-3, Name "Parameters/W"    ];
a    = DefineNumber[  25.4e-3, Name "Parameters/a"    ];
aw   = DefineNumber[    1e-12, Name "Parameters/aw"   ];
beta = DefineNumber[       60, Name "Parameters/beta" ];
// Spatial step
lc = DefineNumber[ 0.4e-3, Name "Parameters/lc"];
lcmax = 25*lc;

//// Points
// Bot
Point(11) = {-W/2, -L/2, 0, lcmax};
Point(12) = {+W/2, -L/2, 0, lcmax};
// Bot
Point(21) = {-W/2, +L/2, 0, lcmax};
Point(22) = {+W/2, +L/2, 0, lcmax};
// Mid point right
Point(31) = { W/2, +0.8*a*Sin(Pi/2-beta*Pi/180), 0, lcmax};
// Crack
Point(32) = {+a*Cos(Pi/2-beta*Pi/180), +a*Sin(Pi/2-beta*Pi/180), 0, lcmax};
Point(33) = {-a*Cos(Pi/2-beta*Pi/180), -a*Sin(Pi/2-beta*Pi/180), 0, lcmax};
// Mid point left
Point(34)  = {-W/2, -0.8*a*Sin(Pi/2-beta*Pi/180), 0, lcmax};

//// Lines
// Bot part
Line(11) = {11, 12};
Line(12) = {12, 31};
Line(13) = {31, 32};
Line(14) = {32, 33}; // Crack line bot
Line(15) = {33, 34};
Line(16) = {34, 11};
// Top part
Line(21) = {21, 22};
Line(22) = {22, 31};
// Line(13)
Line(24) = {32, 33}; // Crack line top
// Line(15)
Line(26) = {34, 21};

//// Surfaces
// Bot
Curve Loop(1) = {11, 12, 13, 14, 15, 16};
Plane Surface(1) = {1};
// Top
Curve Loop(2) = {21, 22, 13, 24, 15, 26};
Plane Surface(2) = {2};

//// Physical groups
// Domain
Physical Surface("domain", 11) = {1, 2};
// Boundaries
Physical Curve("bot", 9) = {11};
Physical Curve("top", 10) = {21};
// Crack
Physical Curve("crack", 12) = {14,24};


////
// Element size
/////////////////
// Create a distance field
Field[1] = Distance;
Field[1].CurvesList = {13, 14, 24, 15};
Field[1].Sampling = 100;
// Use a distance field and a threshold to set the element size
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = a/4;
Field[2].DistMax = 1.5*a;
Field[2].SizeMin = lc;
Field[2].SizeMax = lcmax;
// Set the treshold field as the background field
Background Field = 2;

