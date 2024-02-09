//// Parameters
// Numeric
lc = DefineNumber[ 0.01, Name "Parameters/lc" ];
// Geometry
W  = DefineNumber[    1.0, Name "Parameters/W" ];
a0  = DefineNumber[   0.5, Name "Parameters/a0" ];

//// Points
// Bot
Point(11) = {  0,   0, 0, lc};
Point(12) = {  W,   0, 0, lc};
Point(13) = {  W, W/2, 0, lc};
Point(14) = { a0, W/2, 0, lc}; // Pre-crack tip
Point(15) = {  0, W/2, 0, lc}; // Pre-crack bot lip
// Top
Point(21) = {  0,   W, 0, lc};
Point(22) = {  W,   W, 0, lc};
// Point(13)
// Point(14)
Point(25) = {  0, W/2, 0, lc}; // Pre-crack bot lip

//// Lines
// Bot
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 11};
// Top
Line(21) = {21, 22};
Line(22) = {22, 13};
// Line(13) = {13, 14};
Line(24) = {14, 25};
Line(25) = {25, 21};

//// Surfaces
// Bot
Curve Loop(1) = {11, 12, 13, 14, 15};
Plane Surface(1) = {1};
// Top
Curve Loop(2) = {21, 22, 13, 24, 25};
Plane Surface(2) = {2};

//// Physical groups
// Domain
Physical Surface("domain", 1) = {1, 2};
// Pins
Physical Curve("bot", 2) = {11};
Physical Curve("top", 3) = {21};
// Crack physical groups
Physical Curve("crack", 10) = {14, 24};

