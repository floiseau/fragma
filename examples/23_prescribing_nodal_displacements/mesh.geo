//// Paramters
lc = DefineNumber[ 0.01, Name "Parameters/lc" ];
//// Points
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};
//// Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
//// Surfaces
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
//// Physic groups
Physical Surface("domain", 1) = {1};
Physical Curve("top", 10) = {3};
