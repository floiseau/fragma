//// Options
Mesh.Algorithm = 5;

//// Parameters
// Mechanical
ell = 0.015e-3;
// Geometry
L = 1e-3;
// Numerical
ell_over_h = DefineNumber[ 6, Name "Parameters/ell_over_h" ];
h_min = ell/ell_over_h;
h = 4*ell;

//// Points
// Bot
Point(11) = {0, 0, 0, h};
Point(13) = {L, 0, 0, h};
// Mid (bot)
Point(21) = {0, L/2-h_min/2, 0, h};
Point(22) = {L/2, L/2-h_min/2, 0, h};
// Mid (top)
Point(31) = {0, L/2+h_min/2, 0, h};
Point(32) = {L/2, L/2+h_min/2, 0, h};
// Top
Point(41) = {0, L, 0, h};
Point(43) = {L, L, 0, h};

//// Lines
Line(1) = {11, 13};
Line(2) = {13, 43};
Line(3) = {43, 41};
Line(4) = {41, 31};
Line(5) = {31, 32};
Line(6) = {32, 22};
Line(7) = {22, 21};
Line(8) = {21, 11};

//// Surfaces
// Bottom part
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

//// Physical groups
// Domain
Physical Surface("domain", 1) = {1};
// Lines
Physical Curve("bot", 11) = {1};
Physical Curve("top", 12) = {3};
Physical Curve("crack", 13) = {5, 6, 7};

//// Element size
// Create line for refinement
Point(101) = {0, L/2, 0, h};
Point(102) = {L, L/2, 0, h};
Line(101) = {101, 102};
// Create a distance field
Field[1] = Distance;
Field[1].CurvesList = {101};
Field[1].Sampling = 100;
// Use a distance field and a threshold to set the element size
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].DistMin = 3*ell;
Field[2].DistMax = 8*ell;
Field[2].SizeMin = h_min;
Field[2].SizeMax = h;
// Set the treshold field as the background field
Background Field = 2;
