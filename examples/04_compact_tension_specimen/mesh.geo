//// Options ///////////////////////////
SetFactory("OpenCASCADE");
Mesh.Algorithm = 5;

//// Parameters ////////////////////////
// Numerical
ell = DefineNumber[ 10e-3, Name "Parameters/ell" ]; 
h = DefineNumber[ ell/8, Name "Parameters/h" ];
hmax = DefineNumber[ 16*h, Name "Parameters/hmax" ];
// Boundaries
W =  DefineNumber[ 1.0, Name "Parameters/W" ];
H =  1.2*W;
// Pin holes
phh = 0.325*W;
D = 0.25*W;
// Crack
a0 = DefineNumber[ 0.1*W, Name "Parameters/a0" ];

//// Points ////////////////////////////
// Boundary
Point(1) = {-D, 0, 0, hmax};
Point(2) = { W, 0, 0, hmax};
Point(3) = { W, H, 0, hmax};
Point(4) = {-D, H, 0, hmax};
// Crack
Point(5) = {-D, H/2+h/2, 0, hmax};
Point(6) = {a0, H/2+h/2, 0, hmax};
Point(7) = {a0, H/2-h/2, 0, hmax};
Point(8) = {-D, H/2-h/2, 0, hmax};

//// Lines /////////////////////////////
// Boundaries + Notch
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
// Pin Holes
Circle(12) = {0, phh, 0, D/2, 0, 2*Pi};
Circle(13) = {0, H-phh, 0, D/2, 0, 2*Pi};

//// Surfaces //////////////////////////
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Curve Loop(2) = {12};
Curve Loop(3) = {13};
Plane Surface(1) = {1, 2, 3};

//// Physical group ////////////////////
Physical Surface("Domain", 20) = {1};
Physical Curve("crack", 21) = {5, 6, 7};
Physical Curve("bot_pin", 22) = {12};
Physical Curve("top_pin", 23) = {13};

//// Element size /////////////////////
// Number of points to discretize circle
Mesh.MinimumCirclePoints = (Pi*D)/hmax;
// Create geometric entities
Point(20) = {a0, H/2, 0, hmax};
Point(21) = {W, H/2, 0, hmax};
Line(20) = {20, 21};
// Distance fields
Field[1] = Distance;
Field[1].CurvesList = {20};
Field[1].Sampling = 100;
// Threshold field
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h;
Field[2].SizeMax = hmax;
Field[2].DistMin = 10*h;
Field[2].DistMax = 20*h;
// Apply field 2 as element size
Background Field = 2;
