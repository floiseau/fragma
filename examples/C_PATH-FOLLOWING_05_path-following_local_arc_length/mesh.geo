//// Options ///////////////////////////
Mesh.Algorithm = 5;

//// Parameters ////////////////////////
// Numerical
ell = DefineNumber[ 2e-2, Name "Parameters/ell" ]; 
h = DefineNumber[ ell/6, Name "Parameters/h" ];
hmax = DefineNumber[ 16*h, Name "Parameters/hmax" ];
// Boundaries
W =  DefineNumber[ 1.0, Name "Parameters/W" ];
H =  1.2*W;
// Pin holes
phh = 0.325*W;
D = 0.25*W;
angular_portion_load = 90 * Pi/180; // Load circle arc: See the thesis of Xinyuan Zhai. 
alpha = (Pi - angular_portion_load) / 2;
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
// Bot pin hole
Point(11) = {0, phh, 0};
Point(12) = {0, phh+D/2, 0};
Point(13) = {-D/2*Cos(alpha), phh - D/2*Sin(alpha), 0};
Point(14) = {D/2*Cos(alpha), phh - D/2*Sin(alpha), 0};
// Top pin hole
Point(21) = {0, H-phh, 0};
Point(22) = {0, H-phh-D/2, 0};
Point(23) = {-D/2*Cos(alpha), H-phh+D/2*Sin(alpha), 0};
Point(24) = {D/2*Cos(alpha), H-phh+D/2*Sin(alpha), 0};

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
// Bot pin hole
Circle(11) = {12, 11, 13};
Circle(12) = {13, 11, 14};
Circle(13) = {14, 11, 12};
// Top pin hole
Circle(21) = {22, 21, 23};
Circle(22) = {23, 21, 24};
Circle(23) = {24, 21, 22};

//// Surfaces //////////////////////////
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Curve Loop(2) = {11, 12, 13};
Curve Loop(3) = {21, 22, 23};
Plane Surface(1) = {1, 2, 3};

//// Physical group ////////////////////
Physical Surface("Domain", 1) = {1};
Physical Curve("bot_pin", 2) = {12};
Physical Curve("top_pin", 3) = {22};
Physical Curve("crack", 4) = {5, 6, 7};

//// Element size /////////////////////
// Number of points to discretize circle
Mesh.MinimumCirclePoints = (Pi*D)/hmax;
// Create geometric entities
Point(101) = {-D, H/2, 0, hmax};
Point(102) = {W, H/2, 0, hmax};
Line(101) = {101, 102};
// Distance fields
Field[1] = Distance;
Field[1].CurvesList = {101};
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
