//// Parameters
// Numeric
lc = DefineNumber[ 0.01, Name "Parameters/lc" ];
// Geometry
W  = DefineNumber[    1.0, Name "Parameters/W" ];
aH  = DefineNumber[ 1e-12, Name "Parameters/aH" ];
a0  = DefineNumber[   0.5, Name "Parameters/a0" ];

//// Points
// Boundary
Point(1) = {0, 0, 0, lc};
Point(2) = {W, 0, 0, lc};
Point(3) = {W, W, 0, lc};
Point(4) = {0, W, 0, lc};
Point(5) = {0, W/2+aH, 0, lc};
Point(6) = {a0, W/2, 0, lc}; // Notch tip
Point(7) = {0, W/2-aH, 0, lc};

//// Lines
// Boundary
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 1};

//// Surfaces
// Boundaries
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};

//// Physical groups
// Domain
Physical Surface("domain", 1) = {1};
// Pins
Physical Curve("bot", 2) = {1};
Physical Curve("top", 3) = {3};
// Crack physical groups
Physical Curve("crack", 10) = {5,6};

