//// Options
Mesh.Algorithm = 5;
//// Parameters
// Mechanical
ell = 0.1;
// Geometry
Ly = 1;
Lx = 4 * Ly;
DefineConstant[omega = {-Pi/12, Name "Parameters/omega"}];
// Numerical
h = ell/6;

//// Points
Point(1) = {0, 0, 0, h};
Point(2) = {Lx, 0, 0, h};
Point(3) = {Lx, Ly, 0, h};
Point(4) = {0, Ly, 0, h};
Point(5) = {0, Ly/2+h/2, 0, h};
Point(6) = {Ly/2 * Cos(omega) - h/2 * Sin(omega), Ly/2 + Ly/2 * Sin(omega) + h/2 * Cos(omega), 0, h};
Point(7) = {Ly/2 * Cos(omega) + h/2 * Sin(omega), Ly/2 + Ly/2 * Sin(omega) - h/2 * Cos(omega), 0, h};
Point(8) = {0, Ly/2-h/2, 0, h};

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
