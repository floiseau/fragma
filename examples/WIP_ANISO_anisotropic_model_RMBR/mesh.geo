//// Options
Mesh.Algorithm = 5;
//// Parameters
// Mechanical
ell = 1e-4;
// Geometry
L = 1e-3;
// Numerical
h = ell/10;
// Crack
DefineConstant[geometric_crack = {1, Name "Parameters/geometric_crack"}];

//// Points
Point(1) = {0, 0, 0, h};
Point(2) = {L, 0, 0, h};
Point(3) = {L, L, 0, h};
Point(4) = {0, L, 0, h};
Point(5) = {0, L/2+h/2, 0, h};
Point(6) = {L/2, L/2+h/2, 0, h};
Point(7) = {L/2, L/2-h/2, 0, h};
Point(8) = {0, L/2-h/2, 0, h};

//// Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};

//// Surfaces
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

//// Physical groups
Physical Surface("domain", 1) = {1};
Physical Curve("bot", 2) = {1};
Physical Curve("top", 3) = {3};
Physical Curve("crack", 4) = {7, 6, 5};
Physical Curve("left_bot", 9) = {8};
Physical Curve("left_top", 10) = {4};
