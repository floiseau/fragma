////
// Options
////////////
SetFactory("Built-in");
Geometry.AutoCoherence = 0;
Mesh.SaveAll = 1;

////
// Parameters
///////////////
// Geometry
L  = DefineNumber[    1.0, Name "Parameters/L" ];
H  = DefineNumber[ 0.28*L, Name "Parameters/H" ];
a  = DefineNumber[  0.5*L, Name "Parameters/a" ];
g  = DefineNumber[  0.1*L, Name "Parameters/g" ];
nw = DefineNumber[  1e-16, Name "Parameters/nw" ];
// Numerical
lc = DefineNumber[ 0.01, Name "Parameters/lc" ];

////
// Points
///////////
// Center part
Point(1) = { -a,    -nw, 0, lc};
Point(2) = { -a, -H/2+g, 0, lc};
Point(3) = {L-a, -H/2+g, 0, lc};
Point(4) = {L-a,  H/2-g, 0, lc};
Point(5) = { -a,  H/2-g, 0, lc};
Point(6) = { -a,     nw, 0, lc};
// Crack tip
Point(7) = {  0,      0, 0, lc};
// TODO Upper grip
// TODO Lower grip

////
// Lines
//////////
// Boundaries
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
// Crack line
Line(6) = {6, 7};
Line(7) = {7, 1};

////
// Surfaces
/////////////
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};

////
// Physical groups
////////////////////
// Volume
Physical Surface("domain", 11) = {1};
// Boundaries
Physical Curve("left_bot", 9) = {1};
Physical Curve("bot", 7) = {2};
Physical Curve("top", 8) = {4};
Physical Curve("left_top", 10) = {5};
// TODO Add grip

