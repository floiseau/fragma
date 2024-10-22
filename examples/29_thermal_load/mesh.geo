////
// Parameters
///////////////
// Geometry
L  = DefineNumber[ 1, Name "Parameters/L" ];
// Numerical
lc = DefineNumber[  0.1, Name "Parameters/lc" ];

////
// Points
///////////
Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, L, 0, lc};
Point(4) = {0, L, 0, lc};

////
// Lines
//////////
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

////
// Surfaces
/////////////
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

////
// Physical groups
////////////////////
// BCs
Physical Curve("bot" , 5) = {1};
Physical Curve("right", 6) = {2};
Physical Curve("top" , 7) = {3};
Physical Curve("left" , 8) = {4};
// Domain
Physical Surface("domain", 9) = {1};

////
// Options
////////////
Mesh.SaveAll = 1;
